import os
import flask

def use_path_which_exists(list_of_possible_paths):
    used_path = ''
    assigned = False

    for path in list_of_possible_paths:
        if os.path.exists(path):
            used_path = path
            assigned = True

    if not assigned:
        print ("Error, cannot locate the path of project, will likely fail!")

    return used_path

def get_data_from_list(crop_per_frames):
    image_paths = []
    frame_ids = []
    crop_ids = []

    for frame_i in range(0,len(crop_per_frames)):
        crops_of_frame = crop_per_frames[frame_i]

        for crop_i in range(0,len(crops_of_frame)):
            crop = crops_of_frame[crop_i]

            full_path = crop[0]
            image_paths.append(full_path)
            frame_ids.append(frame_i)
            crop_ids.append(crop_i)

            #print(frame_i, crop_i, crop)

    # flattened!!!
    ground_truths = None
    return image_paths, ground_truths, frame_ids, crop_ids

def run_on_image(IMAGE, yolo_model, sess):

    from crop_functions import mask_from_one_image

    SETTINGS = {}
    SETTINGS["attention_horizontal_splits"] = 2
    SETTINGS["debug_save_crops"] = False

    mask_crop_folder=''

    # 1 generate crops from full images
    mask_crops_per_frames = []
    scales_per_frames = []
    mask_crops_number_per_frames = []

    mask_crops, scale_full_img, attention_crop_TMP_SIZE_FOR_MODEL = mask_from_one_image(IMAGE,
                                                                                            SETTINGS,
                                                                                            mask_crop_folder)  ### <<< mask_crops
    mask_crops_per_frames.append(mask_crops)
    mask_crops_number_per_frames.append(len(mask_crops))
    scales_per_frames.append(scale_full_img)

    print("")


    crop_data = {}
    crop_data["mask_crops_number_per_frames"] = mask_crops_number_per_frames
    crop_data["mask_crops_per_frames"] = mask_crops_per_frames
    crop_data["attention_crop_TMP_SIZE_FOR_MODEL"] = attention_crop_TMP_SIZE_FOR_MODEL

    direct_images = []
    direct_images.append(IMAGE)

    # 2 eval these
    # calculate
    masks_evaluation_times, masks_additional_times, bboxes_per_frames = run_yolo(crop_data, direct_images, yolo_model, sess)

    return bboxes_per_frames

def run_yolo(crop_data, direct_images, yolo_model, sess):

    num_crops_per_frames = crop_data["mask_crops_number_per_frames"]
    crop_per_frames = crop_data["mask_crops_per_frames"]
    fixbb_crop = crop_data["attention_crop_TMP_SIZE_FOR_MODEL"]

    model_h5 = 'yolo.h5'
    anchors_txt = 'yolo_anchors.txt'
    allowed_number_of_boxes = 100
    VERBOSE = 1
    resize_frames = None


    yolo_paths = ["/home/ekmek/YAD2K/", "/home/vruzicka/storage_pylon2/YAD2K/"]
    path_to_yolo = use_path_which_exists(yolo_paths)

    print (path_to_yolo)

    import sys,site
    site.addsitedir(path_to_yolo)
    print (sys.path)  # Just verify it is there
    import yad2k, eval_yolo, eval_yolo_direct_images, eval_yolo_direct_images_take2

    ################################################################
    num_frames = len(num_crops_per_frames)
    image_names, ground_truths, frame_ids, crop_ids = get_data_from_list(crop_per_frames)
    print (len(image_names), image_names[0:2])

    args = {}

    #model_h5 = 'yolo_832x832.h5'
    args["anchors_path"]=path_to_yolo+'model_data/' + anchors_txt
    args["classes_path"]=path_to_yolo+'model_data/coco_classes.txt'
    args["model_path"]=path_to_yolo+'model_data/' + model_h5
    args["score_threshold"]=0.3
    args["iou_threshold"]=0.5
    args["output_path"]=''
    args["test_path"]=''
    print(args)

    pureEval_times, ioPlusEval_times, bboxes = eval_yolo_direct_images_take2._main_modelless(args, yolo_model, sess, direct_images=direct_images, crops_bboxes=crop_per_frames, crop_value=fixbb_crop, resize_frames=resize_frames, verbose=VERBOSE, person_only=True, allowed_number_of_boxes=allowed_number_of_boxes)

    bboxes_per_frames = []
    for i in range(0,num_frames):
        bboxes_per_frames.append([])

    for index in range(0,len(image_names)):
        frame_index = frame_ids[index] - frame_ids[0]
        crop_index = crop_ids[index]

        if bboxes_per_frames[frame_index] is None:
            bboxes_per_frames[frame_index] = []

        crops_in_frame = crop_per_frames[frame_index]
        current_crop = crops_in_frame[crop_index]

        a_left = current_crop[1][0]
        a_top = current_crop[1][1]
        a_right = current_crop[1][2]
        a_bottom = current_crop[1][3]
        debug_bbox = [['crop',[a_top,a_left,a_bottom,a_right],1.0,70]]

        if len(bboxes[index]) > 0: #not empty
            #fixed_bboxes = []
            for bbox in bboxes[index]:
                bbox_array = bbox[1]
                fix_array = bbox_array + [a_top, a_left, a_top, a_left]
                fix_array = fix_array.tolist()

                bboxes_per_frames[frame_index].append([bbox[0],fix_array,float(bbox[2]),int(bbox[3])])

            #bboxes_per_frames[frame_index] += fixed_bboxes
        bboxes_per_frames[frame_index] += debug_bbox

    return pureEval_times, ioPlusEval_times, bboxes_per_frames




def run_on_single_crop(CROP, yolo_model, sess):

    bboxes = run_yolo_one_crop(CROP, yolo_model, sess)

    return bboxes

def run_yolo_one_crop(crop_img, yolo_model, sess):

    yolo_paths = ["/home/ekmek/YAD2K/", "/home/vruzicka/storage_pylon2/YAD2K/"]
    path_to_yolo = use_path_which_exists(yolo_paths)
    import site
    site.addsitedir(path_to_yolo)
    import yad2k, eval_yolo, eval_yolo_direct_images_take2

    ################################################################
    args = {}
    args["anchors_path"]=path_to_yolo+'model_data/yolo_anchors.txt'
    args["classes_path"]=path_to_yolo+'model_data/coco_classes.txt'
    args["score_threshold"]=0.3
    args["iou_threshold"]=0.5

    direct_images = [crop_img]
    pureEval_times, ioPlusEval_times, bboxes = eval_yolo_direct_images_take2.one_crop_modelless_sessionless(args, yolo_model, sess, direct_images)

    jsonable_bboxes_all = []
    for bboxes_inimg in bboxes:
        jsonable_bboxes = []
        for bbox in bboxes_inimg:
            jsonable = [bbox[0], bbox[1].tolist(), float(bbox[2]),int(bbox[3])]
            jsonable_bboxes.append(jsonable)

        jsonable_bboxes_all.append(jsonable_bboxes)

    return jsonable_bboxes_all
