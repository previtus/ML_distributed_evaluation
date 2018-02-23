# Proper server can have these queues:
# crop_queue - queue of images to be evaluated
# (maybe fullframe_queue - we could theoretically want to have the full 4k images too)
# bboxes_queue - bounding boxes predictions, about to be shipped back

import queue
from timeit import default_timer as timer

crop_queue = queue.Queue()
bboxes_queue = queue.Queue()

# CROPS

def put_crop_to_queue(crop_image, uid):
    time = timer()
    queue_object = [uid, time, crop_image]
    crop_queue.put(queue_object)

    print("Enqueue crop uid", uid)

def put_crops_to_queue(crop_images, uids):
    for i in range(0,len(crop_images)):
        put_crop_to_queue(crop_images[i], uids[i])

def get_crops_from_queue(n,filter_time=None):
    crops = []
    extracted = 0
    while True:
        # experiment with timeout?
        queue_object = None

        try:
            queue_object = crop_queue.get()
        except queue.Empty:
            # Loaded all items there are, exit this even if its less than n
            break

            pass
        else:
            if filter_time is not None:
                time_now = timer()
                time_dif = queue_object[1] - time_now
                if time_dif > filter_time:
                    #skipping this one, its too old
                    continue
            crops.append(queue_object)
            crop_queue.task_done()
            extracted += 1

            if extracted >= n:
                break

    print("Got ",len(crops), "crops from the queue. (Out of",n,"requested).")
    return crops


def get_all_from_queue(filter_time=None):
    crops = []
    while True:
        if crop_queue.empty():
            break

        try:
            queue_object = crop_queue.get()
        except queue.Empty:
            # Loaded all items there are, exit this even if its less than n
            break

            pass
        else:
            if filter_time is not None:
                time_now = timer()
                time_dif = queue_object[1] - time_now
                if time_dif > filter_time:
                    #skipping this one, its too old
                    continue
            crops.append(queue_object)
            crop_queue.task_done()

    if len(crops)>0:
        print("Got ",len(crops), "crops from the queue. (Asked for all)")
    return crops

# BBOXES

def put_bbox_to_queue(bbox, uid, start_time):
    bbox_object = [uid, start_time, bbox]
    bboxes_queue.put(bbox_object)

def put_bboxes_to_queue(bboxes, uids, times):
    for i in range(0, len(bboxes)):
        put_bbox_to_queue(bboxes[i], uids[i], times[i])

def get_all_bboxes_from_queue(filter_time=None):
    bboxes = []
    while True:
        if bboxes_queue.empty():
            break

        try:
            bbox_object = bboxes_queue.get()
        except queue.Empty:
            # Loaded all items there are, exit this even if its less than n
            break

            pass
        else:
            if filter_time is not None:
                time_now = timer()
                time_dif = bbox_object[1] - time_now
                if time_dif > filter_time:
                    #skipping this one, its too old
                    continue
            bboxes.append(bbox_object)
            bboxes_queue.task_done()

    if len(bboxes)>0:
        print("Got ",len(bboxes), "bboxes from the queue. (Asked for all)")
    return bboxes