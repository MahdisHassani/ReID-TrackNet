import cv2

track_history = {}

def draw_boxes(frame, tracks):
    global track_history

    for box, track_id in tracks:
        x1,y1,x2,y2 = map(int, box)

        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)

        if track_id not in track_history:
            track_history[track_id] = []

        track_history[track_id].append((cx, cy))

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        for i in range(1, len(track_history[track_id])):
            cv2.line(frame,
                     track_history[track_id][i-1],
                     track_history[track_id][i],
                     (250,0,0), 1)

    return frame