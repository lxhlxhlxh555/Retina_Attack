### README

##### 项目简介

本仓库用于糖尿病视网膜模型的攻击，分为无模型的攻击和有模型的攻击。

##### 参数解释

```python
#General settings
--pic_dir PIC_DIR     Choose to do adversarial augmentation  
--origin_dir ORIGIN_DIR
                      The direction of origin image
--output_dir OUTPUT_DIR
                      The direction of output image
--attack_type {adversarial,normal}
                      Choose an attack type
  
#Adversarial attack arguments
--device DEVICE       Define model device
--model_dir MODEL_DIR
                      The direction of model
--ground_truth {0,1,2,3,4}
                      The ground truth of the picture
--adv_level {1,2,3}   Choose an attack level

#Normal attack arguments
-gn, --gaussian_noise
                        Add gaussian noise
-gb, --gaussian_blur  Add gaussian blur
-sp, --sp_noise       Add salt and pepper noise
-mb, --motion_blur    Add motion blur
-rgb, --rgb_shift     Add rgb shift
-hsv, --hsv_shift     Add hsv shift
-bc, --brightness_contrast
                      Add brightness contrast change
```

##### 运行方法

无模型攻击运行示例：

```python
python attack.py --attack_type normal --pic_dir ./example.jpeg --origin_dir origin.jpeg --output_dir adv.jpeg -gn -gb -sp -mb -rgb -hsv -bc
```

有模型攻击运行示例：

```
python attack.py --attack_type adversarial --pic_dir ./example.jpeg --origin_dir origin.jpeg --output_dir adv.jpeg --device cpu --model_dir jit_module_cpu.pth --ground_truth 0 --adv_level 1
```

