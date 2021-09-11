# obsClient = ObsClient(
#     access_key_id='0YGLVBGF0NWSRF55MSX5',    
#     secret_access_key='jZ664BogYajhpDBgeMJX9aTJCrIJLdWPOb3EbkjI',    
#     server='https://obs.cn-north-4.myhuaweicloud.com'
# )

# @celery_app.task(bind=True)
# def adv_attack(self,upload_files,level,labels):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     imgs = []
#     # for img_f in upload_files:
#     #     img_f = img_f.read()
#     #     im = cv2.imdecode(np.frombuffer(img_f, np.uint8), cv2.IMREAD_COLOR)
#     #     imgs.append(im)
#     fail_to_read = []
#     fail_to_attack = []
#     for img_f in upload_files:
#         resp = obsClient.getObject('zhongjianyuan',img_f,'./test/{}'.format(img_f.split('/')[-1]))
#         if resp.status < 300:
#             im = cv2.imread('./test/{}'.format(img_f.split('/')[-1]))
#             imgs.append(im)
#             print('Read Success!')
#         else:
#             fail_to_read.append(img_f)
#             print('errorCode:', resp.errorCode)
#             print('errorMessage:', resp.errorMessage)
#     assert(len(labels)==len(imgs))
#     adv_imgs = []
#     if device == 'cpu':
#         model = torch.jit.load('./app/weights/jit_module_448_cpu.pth')
#     else:
#         model = torch.jit.load('./app/weights/jit_module_448_gpu.pth')
#     model = model.to(device)
#     attacker = Mask_PGD(model,device)
#     # attacker = Smooth_PGD(model,device)
#     if level == 1:
#         print('Level:1')
#         eps=0.005
#         iter_eps=0.001
#         nb_iter=20
#     elif level == 2:
#         print('Level:2')
#         eps=0.02
#         iter_eps = 0.002
#         nb_iter = 40
#     elif level == 3:
#         print('Level:3')
#         eps = 0.1
#         iter_eps = 0.003
#         nb_iter = 60
#     for count,img in enumerate(imgs):
#         img = img.astype(np.uint8)
#         img = pad_img(img)
#         h,w,c = img.shape
#         img = cv2.resize(img,(448,448))
#         transform = transforms.ToTensor()
#         reverse = transforms.ToPILImage()
#         img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
#         img = transform(img.astype(np.uint8))
#         img = img.unsqueeze(0)
#         # print(img.shape)
#         print("Start Attacking...")
#         adv_img = attacker.generate(img.to(device),y=torch.tensor([labels[count]]).to(device),eps=eps,iter_eps=iter_eps,nb_iter=nb_iter,mask=None)
#         # adv_img = attacker.generate(img.to(device),y=torch.tensor([labels[count]]).to(device),eps=eps,iter_eps=iter_eps,nb_iter=nb_iter,mask=None,sizes=sizes,sigmas=sigmas)
#         adv_img = torch.tensor(adv_img).to(device)
#         #Save adversarial image
#         img = reverse(adv_img.squeeze())
#         img = img.resize((h,w))
#         adv_img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)

#         timestamp = int(round(time.time() * 1000))
#         outname = upload_files[count].split('/')[-1][:-4] + '_' + str(timestamp) + '.jpg'
#         cv2.imwrite('./test/{}'.format(outname),adv_img)
#         resp = obsClient.putFile('zhongjianyuan','topic4/{}'.format(outname),'./test/{}'.format(outname))
#         if resp.status < 300:
#             print('Upload Success!')
#             message = "Successfully uploaded " + outname
#             self.update_state(state='PROGRESS',
#                               meta={'current':count+1,'success':count+1 -len(fail_to_attack) - len(fail_to_read),'total':len(upload_files),'status':message})
#         else:
#             fail_to_attack.append(upload_files[count])
#             print('errorCode:', resp.errorCode)
#             print('errorMessage:', resp.errorMessage)
#     # cmd = 'rm -rf ./test'
#     # os.system(cmd)
#     return {'code':200,'current':len(upload_files),'success':len(upload_files) - len(fail_to_attack) - len(fail_to_read),'total':len(upload_files),'fail_to_read':fail_to_read,'fail_to_attack':fail_to_attack,'status':'Task completed!'}                

# @celery_app.task(bind=True)
# def norm_attack(self,upload_files,attack_types,attack_levels):
#     imgs = []
#     fail_to_read = []
#     fail_to_attack = []
#     for img_f in upload_files:
#         resp = obsClient.getObject('zhongjianyuan',img_f,'./test/{}'.format(img_f.split('/')[-1]))
#         if resp.status < 300:
#             im = cv2.imread('./test/{}'.format(img_f.split('/')[-1]))
#             imgs.append(im)
#             print('Read Success!')
#         else:
#             fail_to_read.append(img_f)
#             print('errorCode:', resp.errorCode)
#             print('errorMessage:', resp.errorMessage)
#     attack_dict = {}
#     attack_list = []
#     assert(len(attack_types)==len(attack_levels))
#     for i,type in enumerate(attack_types):
#         attack_dict[type] = attack_levels[i]
#     if 'defocus_blur' in attack_types:
#         level_list = [1,3,5,7,9]
#         level = attack_dict['defocus_blur']
#         blur_limit = level_list[int(level)-1]
#         attack_list.append(albumentations.GaussianBlur(blur_limit=blur_limit, p=1))
#     if 'motion_blur' in attack_types:
#         level_list = [10,30,50,70,90]
#         level = attack_dict['motion_blur']
#         blur_limit = level_list[int(level)-1]
#         attack_list.append(albumentations.MotionBlur(blur_limit=blur_limit,p=1))
#     if 'rgb_shift' in attack_types:
#         attack_list.append(albumentations.RGBShift(p=1))
#     if 'hsv_shift' in attack_types:
#         level_list = [5,10,15,20,25]
#         level = attack_dict['hsv_shift']
#         shift_limit = level_list[int(level)-1]
#         attack_list.append(albumentations.HueSaturationValue(hue_shift_limit=shift_limit, sat_shift_limit=shift_limit, val_shift_limit=shift_limit, p=1))
#     if 'brightness_contrast' in attack_types:
#         level_list = [0.1,0.2,0.3,0.4,0.5]
#         level = attack_dict['brightness_contrast']
#         limit = level_list[int(level)-1]
#         attack_list.append(albumentations.RandomBrightnessContrast(brightness_limit=limit, contrast_limit=limit, p=1))
#     album = albumentations.Compose(attack_list)
#     for rank,img in enumerate(imgs):
#         adv_img = album(image=img)["image"]
#         if 'iso_noise' in attack_types:
#             mean_list = [2,5,10,15,20]
#             sigma_list = [30,40,50,60,70]
#             level = attack_dict['iso_noise']
#             idx = int(level) - 1
#             adv_img = add_gaussian_noise(adv_img,[mean_list[idx],sigma_list[idx]],p=1)
#         if 'sp_noise' in attack_types:
#             level_list = [0.9,0.8,0.7,0.6,0.5]
#             level = attack_dict['sp_noise']
#             snr = level_list[int(level)-1]
#             adv_img = add_sp_noise(adv_img,SNR=snr,p=1)

#         timestamp = int(round(time.time() * 1000))
#         outname = upload_files[rank].split('/')[-1][:-4] + '_' + str(timestamp) + '.jpg'
#         cv2.imwrite('./test/{}'.format(outname),adv_img)
#         resp = obsClient.putFile('zhongjianyuan','topic4/{}'.format(outname),'./test/{}'.format(outname))
#         if resp.status < 300:
#             print('Upload Success!')
#             message = "Successfully uploaded " + outname
#             self.update_state(state='PROGRESS',
#                               meta={'current':rank+1,'success':rank+1 -len(fail_to_attack) - len(fail_to_read),'total':len(upload_files),'status':message})
#         else:
#             fail_to_attack.append(upload_files[rank])
#             print('errorCode:', resp.errorCode)
#             print('errorMessage:', resp.errorMessage)
#     return {'code':200,'current':len(upload_files),'success':len(upload_files) - len(fail_to_attack) - len(fail_to_read),'total':len(upload_files),'fail_to_read':fail_to_read,'fail_to_attack':fail_to_attack,'status':'Task completed!'} 


# class NormalAttack(Resource):
#     def get(self):
#         return "Normal Attack"
#     def post(self):
#         options = json.loads(request.data)
#         upload_files = options['imgs']
#         attack_types = options['attack_types']
#         attack_levels = options['attack_levels']
#         # task = norm_attack.apply_async(kwargs={'upload_files': upload_files, 'attack_types': attack_types,'attack_levels':attack_levels})
#         task = norm_attack.apply_async(args=[upload_files,attack_types,attack_levels])
#         print(task.id)
#         return str('Task Id:'+task.id)
#         # return "Normal Attack"

# class QueryNorm(Resource):
#     def post(self,_id):
#         task = norm_attack.AsyncResult(_id)
#         if task.state == 'PENDING':
#             #job did not start yet
#             response = {
#             'state': task.state,
#             'current': 0,
#             'success':0,
#             'total': 1,
#             'status': 'Pending...'
#             }
#         elif task.state != 'FAILURE':
#             response = {
#             'state': task.state,
#             'current': task.info.get('current',2),
#             'success': task.info.get('success', 1),
#             'total': task.info.get('total', 2),
#             'fail_to_read':task.info.get('fail_to_read',[]),
#             'fail_to_attack':task.info.get('fail_to_attack',[]),
#             'status': task.info.get('status', '')
#             }
#             if 'result' in task.info:
#                 response['result'] = task.info['result']
#         else:
#             # something went wrong in the background job
#             response = {
#             'state': task.state,
#             'current': 1,
#             'success':1,
#             'total': 1,
#             'status': str(task.info),  # this is the exception raised
#             }
#         return response

# class AdversarialAttack(Resource):
#     def get(self):
#         return "Adversarial Attack"
#     def post(self):
#         options = json.loads(request.data)
#         upload_files = options['imgs']
#         level = options['adv_level']
#         labels = options['labels']
#         task = adv_attack.apply_async(args=[upload_files,level,labels])
#         print(task.id)
#         return str('Task Id:'+task.id)

# class QueryAdv(Resource):
#     def post(self,_id):
#         task = adv_attack.AsyncResult(_id)
#         if task.state == 'PENDING':
#             #job did not start yet
#             response = {
#             'state': task.state,
#             'current': 0,
#             'success': 0,
#             'total': 1,
#             'status': 'Pending...'
#             }
#         elif task.state != 'FAILURE':
#             response = {
#             'state': task.state,
#             'current': task.info.get('current',0),
#             'success': task.info.get('success', 1),
#             'total': task.info.get('total', 2),
#             'fail_to_read':task.info.get('fail_to_read',[]),
#             'fail_to_attack':task.info.get('fail_to_attack',[]),
#             'status': task.info.get('status', '')
#             }
#             if 'result' in task.info:
#                 response['result'] = task.info['result']
#         else:
#             # something went wrong in the background job
#             response = {
#             'state': task.state,
#             'current': 1,
#             'success':1,
#             'total': 1,
#             'status': str(task.info),  # this is the exception raised
#             }
#         return response

# class Upload(Resource):
#     def post(self,path,filename):
        # post_data = request.files.get('file')
        # upload_files=request.files.getlist('file')
        # print(upload_files)
        # img = post_data.read()
        # # print(img)
        # print(request.files)
        # # print(path)
        # # print(request.data)
        # # print(type(img))
        # im1 = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        # print(im1.shape)
        # if not os.path.exists(path):
        #     os.makedirs(path,exist_ok=True)
        # cv2.imwrite(os.path.join(path,filename),im1)
        # # user_info = request.headers
        # # print(user_info)
        # return 201