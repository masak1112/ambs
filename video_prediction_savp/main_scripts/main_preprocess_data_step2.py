



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="directory containing the processed directories ""boxing, handclapping, handwaving, ""jogging, running, walking")
    parser.add_argument("output_dir", type=str)
    # ML 2020/04/08 S
    # Add vars for ensuring proper normalization and reshaping of sequences
    parser.add_argument("-vars","--variables",dest="variables", nargs='+', type=str, help="Names of input variables.")
    parser.add_argument("-height",type=int,default=64)
    parser.add_argument("-width",type = int,default=64)
    parser.add_argument("-seq_length",type=int,default=20)
    parser.add_argument("-sequences_per_file",type=int,default=2)
    args = parser.parse_args()
    current_path = os.getcwd()
    #input_dir = "/Users/gongbing/PycharmProjects/video_prediction/splits"
    #output_dir = "/Users/gongbing/PycharmProjects/video_prediction/data/era5"
    #partition_names = ['train','val',  'test'] #64,64,3 val has issue#

    ############################################################
    # CONTROLLING variable! Needs to be adapted manually!!!
    ############################################################
    partition = {
            "train":{
           #     "2222":[1,2,3,5,6,7,8,9,10,11,12], # Issue due to month 04, it is missing
                "2010":[1,2,3,4,5,6,7,8,9,10,11,12],
           #     "2012":[1,2,3,4,5,6,7,8,9,10,11,12],
                "2013":[1,2,3,4,5,6,7,8,9,10,11,12],
                "2015":[1,2,3,4,5,6,7,8,9,10,11,12],
                "2019":[1,2,3,4,5,6,7,8,9,10,11,12]
                 },
            "val":
                {"2017":[1,2,3,4,5,6,7,8,9,10,11,12]
                 },
            "test":
                {"2016":[1,2,3,4,5,6,7,8,9,10,11,12]
                 }
            }
    
    # ini. MPI
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()  # rank of the node
    p = comm.Get_size()  # number of assigned nods
  
    if my_rank == 0 :
        # retrieve final statistics first (not parallelized!)
        # some preparatory steps
        stat_dir_prefix = args.input_dir
        varnames        = args.variables
    
        vars_uni, varsind, nvars = get_unique_vars(varnames)
        stat_obj = Calc_data_stat(nvars)                            # init statistic-instance
    
        # loop over whole data set (training, dev and test set) to collect the intermediate statistics
        print("Start collecting statistics from the whole datset to be processed...")
        for split in partition.keys():
            values = partition[split]
            for year in values.keys():
                file_dir = os.path.join(stat_dir_prefix,year)
                for month in values[year]:
                    # process stat-file:
                    stat_obj.acc_stat_master(file_dir,int(month))  # process monthly statistic-file  
        
        # finalize statistics and write to json-file
        stat_obj.finalize_stat_master(vars_uni)
        stat_obj.write_stat_json(args.input_dir)

        # organize parallelized partioning 
        partition_year_month = [] #contain lists of list, each list includes three element [train,year,month]
        partition_names = list(partition.keys())
        print ("partition_names:",partition_names)
        broadcast_lists = []
        for partition_name in partition_names:
            partition_data = partition[partition_name]        
            years = list(partition_data.keys())
            broadcast_lists.append([partition_name,years])
        for nodes in range(1,p):
            #ibroadcast_list = [partition_name,years,nodes]
            #broadcast_lists.append(broadcast_list)
            comm.send(broadcast_lists,dest=nodes) 
           
        message_counter = 1
        while message_counter <= 12:
            message_in = comm.recv()
            message_counter = message_counter + 1 
            print("Message in from slaver",message_in) 
            
        write_sequence_file(args.output_dir,args.seq_length,args.sequences_per_file)
        
        #write_sequence_file   
    else:
        message_in = comm.recv()
        print ("My rank,", my_rank)   
        print("message_in",message_in)
        # open statistics file and feed it to norm-instance
        print("Opening json-file: "+os.path.join(args.input_dir,"statistics.json"))
        with open(os.path.join(args.input_dir,"statistics.json")) as js_file:
            stats = json.load(js_file)
        #loop the partitions (train,val,test)
        for partition in message_in:
            print("partition on slave ",partition)
            partition_name = partition[0]
            save_output_dir =  os.path.join(args.output_dir,partition_name)
            for year in partition[1]:
               input_file = "X_" + '{0:02}'.format(my_rank) + ".pkl"
               temp_file = "T_" + '{0:02}'.format(my_rank) + ".pkl"
               input_dir = os.path.join(args.input_dir,year)
               temp_file = os.path.join(input_dir,temp_file )
               input_file = os.path.join(input_dir,input_file)
               # create the tfrecords-files
               read_frames_and_save_tf_records(year=year,month=my_rank,stats=stats,output_dir=save_output_dir, \
                                               input_file=input_file,temp_input_file=temp_file,vars_in=args.variables, \
                                               partition_name=partition_name,seq_length=args.seq_length, \
                                               height=args.height,width=args.width,sequences_per_file=args.sequences_per_file)   
                                                  
            print("Year {} finished",year)
        message_out = ("Node:",str(my_rank),"finished","","\r\n")
        print ("Message out for slaves:",message_out)
        comm.send(message_out,dest=0)
        
    MPI.Finalize()        
   
if __name__ == '__main__':
     main()

