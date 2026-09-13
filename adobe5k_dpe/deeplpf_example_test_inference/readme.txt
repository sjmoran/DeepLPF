Enhanced by DeepLPF checkpoint:
  pretrained_models/adobe_dpe/deeplpf_psnr_24.180_ssim_0.917_model.pt

run over ../deeplpf_example_test_input/ and scored against
../deeplpf_example_test_output/ by this repository's own inference path
(main.py --checkpoint_filepath --inference_img_dirpath). The PSNR/SSIM in each
filename is that image's score from this run; they average 25.833 dB / 0.915.
Small last-decimal differences are expected when re-running on other hardware.
