from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "ResnetModelFile.h5"))
detector.loadModel()
detections2 = detector.detectObjectsFromImage(
    input_image=os.path.join(execution_path , "./images/image2.jpg"), 
    output_image_path=os.path.join(execution_path , "./newImages/image2lowProb.jpg"),
    minimum_percentage_probability = 30)

for eachObject in detections2:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )