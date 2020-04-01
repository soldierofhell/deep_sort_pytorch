import numpy as np
import cv2

COLORS_10 =[(144,238,144),(178, 34, 34),(221,160,221),(  0,255,  0),(  0,128,  0),(210,105, 30),(220, 20, 60),
            (192,192,192),(255,228,196),( 50,205, 50),(139,  0,139),(100,149,237),(138, 43,226),(238,130,238),
            (255,  0,255),(  0,100,  0),(127,255,  0),(255,  0,255),(  0,  0,205),(255,140,  0),(255,239,213),
            (199, 21,133),(124,252,  0),(147,112,219),(106, 90,205),(176,196,222),( 65,105,225),(173,255, 47),
            (255, 20,147),(219,112,147),(186, 85,211),(199, 21,133),(148,  0,211),(255, 99, 71),(144,238,144),
            (255,255,  0),(230,230,250),(  0,  0,255),(128,128,  0),(189,183,107),(255,255,224),(128,128,128),
            (105,105,105),( 64,224,208),(205,133, 63),(  0,128,128),( 72,209,204),(139, 69, 19),(255,245,238),
            (250,240,230),(152,251,152),(  0,255,255),(135,206,235),(  0,191,255),(176,224,230),(  0,250,154),
            (245,255,250),(240,230,140),(245,222,179),(  0,139,139),(143,188,143),(255,  0,  0),(240,128,128),
            (102,205,170),( 60,179,113),( 46,139, 87),(165, 42, 42),(178, 34, 34),(175,238,238),(255,248,220),
            (218,165, 32),(255,250,240),(253,245,230),(244,164, 96),(210,105, 30)]


def draw_bbox(img, box, cls_name, identity=None, offset=(0,0)):
    '''
        draw box of an id
    '''
    x1,y1,x2,y2 = [int(i+0.0) for idx,i in enumerate(box)] # offset[idx%2]
    # set color and label text
    color = COLORS_10[identity%len(COLORS_10)] if identity is not None else COLORS_10[0]
    label = '{} {}'.format(cls_name, identity, match_method)
    # box text and bar
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    cv2.rectangle(img,(x1, y1),(x2,y2),color,2)
    cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
    cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 1)
    return img


def draw_bboxes(frame_id, new_sequence, img, bbox, identities=None, match_method=None, number=None, number_box=None, detection_id=None, min_cost=None, offset=(0,0)):
    sequence_label = "NEW SEQUENCE" if new_sequence else ""
    cv2.putText(img, f'FRAME_ID: {frame_id} | {sequence_label}', (0, 50), cv2.FONT_HERSHEY_PLAIN, 3, [255,255,255], 2)
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0    
        color = COLORS_10[id%len(COLORS_10)]
        match_dict = {0: 'N', 1: 'F', 2: 'I'}  
        label = '{}{:d}|{}|{}|{}'.format("", id, number[i], detection_id[i], min_cost[i]) # match_dict[match_method[i]]
        label = '{:d}'.format(id)           
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        #if np.max(number_box)>0:
        #    cv2.rectangle(img,(x1+number_box[i,0], y1 + number_box[i,1]),(x1+number_box[i,2], y1 + number_box[i,3]),color,3)
        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
    return img

def softmax(x):
    assert isinstance(x, np.ndarray), "expect x be a numpy array"
    x_exp = np.exp(x*5)
    return x_exp/x_exp.sum()

def softmin(x):
    assert isinstance(x, np.ndarray), "expect x be a numpy array"
    x_exp = np.exp(-x)
    return x_exp/x_exp.sum() 

import configparser
import json
import os

def draw_offline(img_dir, track_json, detection_json, sequence_json, config_yml):
            
    # TODO: tracking_config.ini    

    config = configparser.ConfigParser()
    config.read(config_yml)
    
    with open(track_json) as f:
      track_dict = json.load(f)
    with open(detection_json) as f:
      detection_dict = json.load(f)
    with open(sequence_json) as f:
      sequence_dict = json.load(f)
    
    if config['output']['video']:
        vw = cv2.VideoWriter(config['output']['video'], cv2.VideoWriter_fourcc(*'MJPG'), int(config['output']['fps']), (int(config['output']['width']), int(config['output']['height'])))
    
    # TODO: remove tracks that were Tentative + Deleted

    image_list = sorted(os.listdir(img_dir))[:10]   #  detection_dict.keys()    

    for idx, img_file in enumerate(image_list):       
 
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)

        if config['pitch_projection'].getboolean('show'):
            pitch_img = cv2.imread(config['pitch_projection']['pitch_image'])
            pitch_img = pitch_img[50:-50,50:-50]
            pitch_width = config['pitch_projection'].getint('pitch_width')
            pitch_height = config['pitch_projection'].getint('pitch_height')
            pitch_img = cv2.resize(pitch_img, (pitch_width,pitch_height))
            img[:pitch_height,:pitch_width] = pitch_img

        if config['flags'].getboolean('frame_id'):
            main_label = f'FRAME_ID: {idx+1} | {img_file}'
            if config['flags'].getboolean('frame_id'):
                main_label += '| '
            cv2.putText(img, main_label, (0, 50), cv2.FONT_HERSHEY_PLAIN, 3, [255,255,255], 2)           

        tracks = track_dict[img_path] # track_dict[img_file] # track_dict = {'img_file': []}

        detections = detection_dict[img_path]
            


        #print(tracks)
        
        if config['flags'].getboolean('player_box'):
            if config['player_box'].getboolean('raw_detection'):                
                for detection in detections:
                    detection_id = detection['detection_id']
                    color = COLORS_10[detection_id%len(COLORS_10)]
                    x1,y1,x2,y2 = detection['bbox']            
                    cv2.rectangle(img, (x1,y1), (x2,y2), color, 3)            
                    if config['flags'].getboolean('detection_id'):
                        confidence = detection['confidence']
                        label = f'{detection_id}:{confidence:.2f}' 
                        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
                        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
                        
                    if config['pitch_projection'].getboolean('show'):
                        x, y = detection['coordinates']
                        cv2.circle(img, (int(pitch_width*x),int(pitch_height*y)), 5, color, -1)
                        

                    
            if config['player_box'].getboolean('kalman_box') or config['player_box'].getboolean('detection_box'): 
                for track in tracks:            
                    track_id = track['track_id']           
                    color = COLORS_10[track_id%len(COLORS_10)] # todo:

                    if config['player_box'].getboolean('kalman_box'):
                        box_type = 'kalman_box'
                    else:
                        box_type = 'detection_box'
            
                    if track['time_since_update'] == 0:
                        x1,y1,x2,y2 = track[box_type]            
                        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
                        if config['flags'].getboolean('sequence_override'):
                            number_in_sequence = sequence_dict[track_id]['number']
                            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
                            cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)

                        if config['flags'].getboolean('number'):
                            label = str(track['number'])
                            #print(label)
                            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
                            cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)


                        if config['flags'].getboolean('track_id'):
                            label = str(track['track_id'])
                            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
                            cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
                            
                        if config['pitch_projection'].getboolean('show'):
                            x, y = track['detection_coordinates']
                            cv2.circle(img, (int(pitch_width*x),int(pitch_height*y)), 5, color, -1)
                
                        
                
        if config['output']['video']:
            vw.write(img)
            
    if config['output']['video']:
        vw.release()


if __name__ == '__main__':
    x = np.arange(10)/10.
    x = np.array([0.5,0.5,0.5,0.6,1.])
    y = softmax(x)
    z = softmin(x)
    import ipdb; ipdb.set_trace()
