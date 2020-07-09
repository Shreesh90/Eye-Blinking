import cv2
import dlib
import numpy as np 

def midpt(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def length(diff1, diff2):
    return np.sqrt( diff1**2 + diff2**2)

def eye_blink_ratio(landmarks, points):
    leftpt  = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
    rightpt = (landmarks.part(points[1]).x, landmarks.part(points[1]).y)
    centre_top = midpt(landmarks.part(points[2]), landmarks.part(points[3]))
    centre_bottom = midpt(landmarks.part(points[4]), landmarks.part(points[5]))
    
    cv2.circle(frame, leftpt, 2, (0,0,255), -1)
    cv2.circle(frame, rightpt, 2, (0,0,255), -1)
    cv2.circle(frame, centre_top, 2, (0,0,255), -1)
    cv2.circle(frame, centre_bottom, 2, (0,0,255), -1)
    
    hor_line = cv2.line(frame, leftpt, rightpt, (0,255,0), 1)
    ver_line = cv2.line(frame, centre_top, centre_bottom, (0,255,0), 1)

    hor_len = length(leftpt[0]-rightpt[0], leftpt[1]-rightpt[1])
    ver_len = length(centre_top[0]-centre_bottom[0], centre_top[1]-centre_bottom[1])
    
    ratio = hor_len/ver_len
    return ratio


cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

both_frames=0
left_frames=0
right_frames=0

while cap.isOpened():
    _, frame = cap.read()
    
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(imgray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        
        # cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
        landmarks = predictor(imgray, face)
        # cv2.circle(frame, (landmarks.part(36).x, landmarks.part(36).y), 2, (0,0,255), -1)
        
        ratio_right = eye_blink_ratio(landmarks, [36, 39, 37, 38, 40,41])
        ratio_left = eye_blink_ratio(landmarks, [42, 45, 43, 44, 46,47])
        
        ratio = (ratio_left + ratio_right)/2
        
        
        
        # print(ratio_left, "       ", ratio_right)
        # print(ratio)
        if ratio > 5.8:
            both_frames += 1
            if both_frames==1:
                cv2.putText(frame, 'BLINKED',(50, 50),  cv2.FONT_HERSHEY_DUPLEX , 2, (0,0,255), 10)
                both_frames=0
       
        if ratio_left>3.8 and ratio_right < 3.68:
            left_frames += 1
            if left_frames==2:
                cv2.putText(frame, 'LEFT BLINKED',(50, 50),  cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255), 10)
                left_frames=0
        if ratio_right>3.8 and ratio_left<3.68:
            right_frames += 1
            if right_frames==2:
                cv2.putText(frame, 'RIGHT BLINKED',(50, 50),  cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255), 10)
                right_frames=0
        
        
    cv2.imshow('LIVE', frame)
    if cv2.waitKey(1)&0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



