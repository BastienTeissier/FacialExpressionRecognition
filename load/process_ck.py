import os
import shutil

label_path = "../../Emotion_labels/Emotion/"
image_path = "../../cohn-kanade-images/"

emotions = {1: "angry", 2: "contempt", 3: "disgust", 4: "fear", 5: "happy", 6: "sad", 7: "surprise"}
counter = {"angry":0, "disgust":0, "fear":0, "happy":0, "sad":0, "surprise":0, "neutral":0, "contempt":0}
 
def copyFile(src, dest):
	try:
		shutil.copy(src, dest)
	# eg. src and dest are the same file
	except shutil.Error as e:
		print('Error: %s' % e)
	# eg. source or destination doesn't exist
	except IOError as e:
		print('Error: %s' % e.strerror)

def create_folder(original_path, label):
	counter[label] += 1
	new_folder_path = "result/" + label + "_" + str(counter[label])
	os.makedirs(new_folder_path)
	copyFile(original_path, new_folder_path)

def create_folder_from_label_index(sequence, folder, label, index):
	original_path = image_path + str(sequence) + "/" + str(folder) + "/" + images[index]
	create_folder(original_path, label)

for sequence in os.listdir(label_path):
	for folder in os.listdir(label_path + str(sequence)):
		for file in os.listdir(label_path + str(sequence) + "/" + str(folder)):
			path = label_path + str(sequence) + "/" + str(folder) + "/" + str(file)
			label_file = open(path, 'r')
			label = label_file.read()
			emotion_code = int(label.split()[0][0])
			emotion_label = emotions[emotion_code]
			
			images = os.listdir(image_path + str(sequence) + "/" + str(folder))

			if (len(images) > 3):
				create_folder_from_label_index(sequence, folder, "neutral", 0)
				for index in range(-3, 0):
					create_folder_from_label_index(sequence, folder, emotion_label, index)	
			else:
				print("Pas assez d'images : " + file)