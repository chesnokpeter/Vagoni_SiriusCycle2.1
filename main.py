from ultralytics import YOLO

model = YOLO("best.pt") 

results = model.predict("testdatasets/*/*.jpg", device='cuda') 

for result in results:
    boxes = result.boxes 
    masks = result.masks
    keypoints = result.keypoints
    probs = result.probs 
    obb = result.obb  
    # print(result.path)
    result.save(filename=str(result.path).replace('testdatasets', 'donedatasets'))  

