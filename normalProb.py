from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "ResnetModelFile.h5"))
detector.loadModel()
detections1 = detector.detectObjectsFromImage(
    input_image=os.path.join(execution_path , "./images/image.jpeg"), 
    output_image_path=os.path.join(execution_path , "./newImages/image1new.jpg"),
    minimum_percentage_probability = 50)

detections2 = detector.detectObjectsFromImage(
    input_image=os.path.join(execution_path , "./images/image2.jpg"), 
    output_image_path=os.path.join(execution_path , "./newImages/image2new.jpg"),
    minimum_percentage_probability = 50)

detections3 = detector.detectObjectsFromImage(
    input_image=os.path.join(execution_path , "./images/image3.jpeg"), 
    output_image_path=os.path.join(execution_path , "./newImages/image3new.jpg"),
    minimum_percentage_probability = 50)

detections4 = detector.detectObjectsFromImage(
    input_image=os.path.join(execution_path , "./images/image4.jpg"), 
    output_image_path=os.path.join(execution_path , "./newImages/image4new.jpg"),
    minimum_percentage_probability = 50)

#for eachObject in detections2:
#    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )