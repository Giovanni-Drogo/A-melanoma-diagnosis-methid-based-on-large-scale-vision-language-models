#对应三个不同的随机种子——ours_model
# python -u test.py --prompt_path '/home/wenyu/Clip_prompt/sava/seed1/prompt_model_89.pt' --clip_path '/home/wenyu/Clip_prompt/sava/seed1/Image_encoder_tuning89.pt';
# python -u test.py --prompt_path '/home/wenyu/Clip_prompt/sava/seed2/prompt_model_112.pt' --clip_path '/home/wenyu/Clip_prompt/sava/seed2/Image_encoder_tuning112.pt';
# python -u test.py --prompt_path '/home/wenyu/Clip_prompt/sava/seed3/prompt_model_78.pt' --clip_path '/home/wenyu/Clip_prompt/sava/seed3/Image_encoder_tuning78.pt';


# python -u test_train.py --prompt_path '/home/wenyu/Clip_prompt/sava/seed1/prompt_model_89.pt' --clip_path '/home/wenyu/Clip_prompt/sava/seed1/Image_encoder_tuning89.pt';
# python -u test_train.py --prompt_path '/home/wenyu/Clip_prompt/sava/seed2/prompt_model_112.pt' --clip_path '/home/wenyu/Clip_prompt/sava/seed2/Image_encoder_tuning112.pt';
# python -u test_train.py --prompt_path '/home/wenyu/Clip_prompt/sava/seed3/prompt_model_78.pt' --clip_path '/home/wenyu/Clip_prompt/sava/seed3/Image_encoder_tuning78.pt';

#消融：Baseline+ift
# python -u test.py --prompt_path '/home/wenyu/Clip_prompt/sava/wocg/seed1/prompt_model_34.pt' --clip_path '/home/wenyu/Clip_prompt/sava/wocg/seed1/Image_encoder_tuning34.pt';
# python -u test.py --prompt_path '/home/wenyu/Clip_prompt/sava/wocg/seed2/prompt_model_31.pt' --clip_path '/home/wenyu/Clip_prompt/sava/wocg/seed2/Image_encoder_tuning31.pt';
# python -u test.py --prompt_path '/home/wenyu/Clip_prompt/sava/wocg/seed3/prompt_model_45.pt' --clip_path '/home/wenyu/Clip_prompt/sava/wocg/seed3/Image_encoder_tuning45.pt';
#消融：Baseline+cg
# python -u test.py --prompt_path '/home/wenyu/Clip_prompt/sava/woift_new/seed1/prompt_model_62.pt' --clip_path '' --image_enc 0;
# python -u test.py --prompt_path '/home/wenyu/Clip_prompt/sava/woift_new/seed2/prompt_model_54.pt' --clip_path '' --image_enc 0;
# python -u test.py --prompt_path '/home/wenyu/Clip_prompt/sava/woift_new/seed3/prompt_model_42.pt' --clip_path '' --image_enc 0;
#消融:Baseline：coop
# python -u test.py --prompt_path '/home/wenyu/Clip_prompt/sava/wo_cg_ift/seed1/prompt_model_33.pt' --clip_path '' --image_enc 0;
# python -u test.py --prompt_path '/home/wenyu/Clip_prompt/sava/wo_cg_ift/seed2/prompt_model_24.pt' --clip_path '' --image_enc 0;
# python -u test.py --prompt_path '/home/wenyu/Clip_prompt/sava/wo_cg_ift/seed3/prompt_model_30.pt' --clip_path '' --image_enc 0;



#对应三个不同的随机种子——ours_model
# python -u test_v2.py --prompt_path '/home/wenyu/Clip_prompt/sava/seed1/prompt_model_89.pt' --clip_path '/home/wenyu/Clip_prompt/sava/seed1/Image_encoder_tuning89.pt' --dataset 'ph2';
# python -u test_v2.py --prompt_path '/home/wenyu/Clip_prompt/sava/seed2/prompt_model_112.pt' --clip_path '/home/wenyu/Clip_prompt/sava/seed2/Image_encoder_tuning112.pt' --dataset 'ph2';
# python -u test_v2.py --prompt_path '/home/wenyu/Clip_prompt/sava/seed3/prompt_model_78.pt' --clip_path '/home/wenyu/Clip_prompt/sava/seed3/Image_encoder_tuning78.pt' --dataset 'ph2';


#消融：Baseline+ift
# python -u test_v2.py --prompt_path '/home/wenyu/Clip_prompt/sava/wocg/seed1/prompt_model_34.pt' --clip_path '/home/wenyu/Clip_prompt/sava/wocg/seed1/Image_encoder_tuning34.pt' --dataset 'ph2';
# python -u test_v2.py --prompt_path '/home/wenyu/Clip_prompt/sava/wocg/seed2/prompt_model_31.pt' --clip_path '/home/wenyu/Clip_prompt/sava/wocg/seed2/Image_encoder_tuning31.pt' --dataset 'ph2';
# python -u test_v2.py --prompt_path '/home/wenyu/Clip_prompt/sava/wocg/seed3/prompt_model_45.pt' --clip_path '/home/wenyu/Clip_prompt/sava/wocg/seed3/Image_encoder_tuning45.pt' --dataset 'ph2';

#消融：Baseline+cg
# python -u test_v2.py --prompt_path '/home/wenyu/Clip_prompt/sava/woift_new/seed1/prompt_model_62.pt' --clip_path '' --image_enc 0  --dataset 'ph2';
# python -u test_v2.py --prompt_path '/home/wenyu/Clip_prompt/sava/woift_new/seed2/prompt_model_54.pt' --clip_path '' --image_enc 0  --dataset 'ph2';
# python -u test_v2.py --prompt_path '/home/wenyu/Clip_prompt/sava/woift_new/seed3/prompt_model_42.pt' --clip_path '' --image_enc 0  --dataset 'ph2';

#消融:Baseline：coop
python -u test_v2.py --prompt_path '/home/wenyu/Clip_prompt/sava/wo_cg_ift/seed1/prompt_model_33.pt' --clip_path '' --image_enc 0  --dataset 'ph2';
python -u test_v2.py --prompt_path '/home/wenyu/Clip_prompt/sava/wo_cg_ift/seed2/prompt_model_24.pt' --clip_path '' --image_enc 0  --dataset 'ph2';
python -u test_v2.py --prompt_path '/home/wenyu/Clip_prompt/sava/wo_cg_ift/seed3/prompt_model_30.pt' --clip_path '' --image_enc 0  --dataset 'ph2';

