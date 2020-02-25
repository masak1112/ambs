import pickle
import os
from matplotlib.pylab import plt

# results_path = ["results_test_samples/era5_size_64_64_3_norm_dup_pretrained/ours_savp","results_test_samples/era5_size_64_64_3_norm_dup_pretrained_finetune/ours_savp",
#                "results_test_samples/era5_size_64_64_3_norm_dup_pretrained_gan/kth_ours_gan","results_test_samples/era5_size_64_64_3_norm_dup_pretrained_vae_l1/kth_ours_vae_l1"]
#
# model_names = ["SAVP","SAVP_Finetune","GAN","VAE"]


# results_path = ["results_test_samples/era5_size_64_64_3_norm_dup_pretrained/ours_savp","results_test_samples/era5_size_64_64_3_norm_msl_gph_pretrained_savp/ours_savp",
#                "results_test_samples/era5_size_64_64_3_norm_dup_pretrained_gan/kth_ours_gan","results_test_samples/era5_size_64_64_3_norm_msl_gph_pretrained_gan/kth_ours_gan"]
#
# model_names = ["SAVP_3T","SAVP_T-MSL-GPH","GAN_3T","GAN_T-MSL_GPH"]
#
# results_path = ["results_test_samples/era5_size_64_64_3_norm_dup_pretrained/ours_savp","results_test_samples/era5_size_64_64_3_norm_dup/ours_savp",
#                 "results_test_samples/era5_size_64_64_3_norm_dup_pretrained/kth_ours_gan","results_test_samples/era5_size_64_64_3_norm_dup/ours_gan",
#                 "results_test_samples/era5_size_64_64_3_norm_dup_pretrained/kth_ours_vae_l1","results_test_samples/era5_size_64_64_3_norm_dup/ours_vae_l1"]
# model_names = ["TF-SAVP(KTH)","SAVP (3T)","TF-GAN(KTH)","GAN (3T)","TF-VAE (KTH)","VAE (3T)"]


results_path = ["results_test_samples/era5_size_64_64_3_norm_t_msl_gph/ours_savp", "results_test_samples/era5_size_64_64_3_norm_dup/ours_savp",
                "results_test_samples/era5_size_64_64_3_norm_t_msl_gph/ours_gan","results_test_samples/era5_size_64_64_3_norm_dup/ours_gan"]
model_names = ["SAVP(T-MSL-GPH)", "SAVP (3T)", "GAN (T-MSL-GPH)","GAN (3T)"]


mse_all = []
psnr_all = []
ssim_all = []
for path in results_path:
    p = os.path.join(path,"results.pkl")
    result = pickle.load(open(p,"rb"))
    mse = result["mse"]
    psnr = result["psnr"]
    ssim = result["ssim"]
    mse_all.append(mse)
    psnr_all.append(psnr)
    ssim_all.append(ssim)


def get_metric(metrtic):
    if metric == "mse":
        return mse_all
    elif metric == "psnr":
        return psnr_all
    elif metric == "ssim":
        return ssim_all
    else:
        raise("Metric error")

for metric in ["mse","psnr","ssim"]:
    evals = get_metric(metric)
    timestamp = list(range(1,11))
    fig = plt.figure()
    plt.plot(timestamp, evals[0],'-.',label=model_names[0])
    plt.plot(timestamp, evals[1],'--',label=model_names[1])
    plt.plot(timestamp, evals[2],'-',label=model_names[2])
    plt.plot(timestamp, evals[3],'--.',label=model_names[3])
    # plt.plot(timestamp, evals[4],'*-.',label=model_names[4])
    # plt.plot(timestamp, evals[5],'--*',label=model_names[5])
    if metric == "mse":
        plt.legend(loc="upper left")
    else:
        plt.legend(loc = "upper right")
    plt.xlabel("Timestamps")
    plt.ylabel(metric)
    plt.title(metric,fontsize=15)
    plt.savefig(metric + "2.png")
    plt.clf()




