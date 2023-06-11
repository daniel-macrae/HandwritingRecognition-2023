from PIL import Image, ImageFont, ImageDraw, ImageChops
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import math
import cv2

from .commonAug import letterImageWarper, imageRotator, imageShearer 


#Character mapping for each of the 27 tokens
char_map = {'Alef' : ')', 
            'Ayin' : '(', 
            'Bet' : 'b', 
            'Dalet' : 'd', 
            'Gimel' : 'g', 
            'He' : 'x', 
            'Het' : 'h', 
            'Kaf' : 'k', 
            'Kaf-final' : '\\', 
            'Lamed' : 'l', 
            'Mem' : '{', 
            'Mem-medial' : 'm', 
            'Nun-final' : '}', 
            'Nun-medial' : 'n', 
            'Pe' : 'p', 
            'Pe-final' : 'v', 
            'Qof' : 'q', 
            'Resh' : 'r', 
            'Samekh' : 's', 
            'Shin' : '$', 
            'Taw' : 't', 
            'Tet' : '+', 
            'Tsadi-final' : 'j', 
            'Tsadi-medial' : 'c', 
            'Waw' : 'w', 
            'Yod' : 'y', 
            'Zayin' : 'z'}

habbakukClasses = list(char_map.keys())


def create_word_page():
    # a list that contains the BB coordinates [x1, y1, x2, y2] for each letter
    BB = []
    CHARACTERS = []
    TEXT = []

    fontSize = random.randint(35, 50)
    font = ImageFont.truetype('data_management/augmentation/Habbakuk.ttf', fontSize)

    widthNoise = 3
    heightNoise = 3
    spacingNoise = [5, 10]
    bbPadding = 0 # how much whitespace on the edges of the BB


    #Create blank image and create a draw interface
    img_size = (1000, 1000)
    img = Image.new('L', img_size, 255) 

    whiteImage = Image.new('L', img_size, 255)  

    draw = ImageDraw.Draw(img)

    rows = np.arange(150,900, random.randint(70, 80))  # rows are 75 pixels apart
    colStart = list(np.arange(200,500,10))
    colEnd = list(np.arange(500,910,10))

    character_set_length = len(list(char_map))

    for row in rows:
        rowText = []
        #for col in cols:
        col = random.choice(colStart)
        endColumn = random.choice(colEnd)

        while col < endColumn:

            if random.random() > 0.2: # 20% chance of no letter = space in the sentence
                # pick a character randomly, get its dimensions
                characterIDX = random.randint(0, character_set_length-1)
                character = list(char_map)[characterIDX]
                symbol = char_map[character]
                w,h = font.getsize(symbol)

                # set its position, with a bit of noise (position is the top-left corner pixel of the letter's BB)
                position = (int(col - (w/2) + random.randint(-widthNoise, widthNoise)),  
                            int(row - (h/2) + random.randint(-heightNoise, heightNoise)) )
                draw.text(position, symbol, 0, font)

                col += w + random.randint(spacingNoise[0], spacingNoise[1])

                CHARACTERS.append(characterIDX)
                BB.append([position[0] - bbPadding, position[1] - bbPadding, position[0]+w + bbPadding, position[1]+h + bbPadding])
                rowText.append(character)
            else:
                col += random.randint(20, 40) # random-sized space between letters
                rowText.append(' ')
        TEXT.append(rowText)



    # to make the image grainy, use this mask. p is the probability that each pixel is made to be white (hence removing parts of the letters)
    p = 0.5
    mask = np.random.rand(img_size[0], img_size[1]) > p
    mask = Image.fromarray(mask)

    img = ImageChops.composite(img, whiteImage, mask) 


    # convert to cv2 image to do morpological operations
    # these operations make the image look much more similar to the dead sea scrolls (in terms of image quality)
    img = np.array(img) 
    
    kernel = np.ones((2, 2), np.uint8) 
    img = cv2.erode(img,kernel,iterations = 2)  # erode the image (makes the black letters wider)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) # apply morphological opening operation, smoothes the edges and removes excess noise

    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU) # make sure the image is binarised


    return img, BB, CHARACTERS, TEXT





def pageImageWarper(img, BB):
    rows, cols = img.shape

    warp_factors = list(np.arange(-40,41,5))
    wave_lengths = list(np.arange(0.9,2.1,0.1))

    horizontalWarpFactor = random.choice(warp_factors)
    verticalWarpFactor = random.choice(warp_factors)
    horzWaveLength = random.choice(wave_lengths)
    vertWaveLength = random.choice(wave_lengths)

    img_output = np.ones(img.shape, dtype=img.dtype) * 255 # make a blank white image to place the warp on

    x_offsets = [int(horizontalWarpFactor * math.sin(2 * 3.14 * x_coord / (cols*horzWaveLength))) for x_coord in np.arange(0,cols,1)] 
    y_offsets = [int(verticalWarpFactor * math.sin(2 * 3.14 * y_coord / (rows*vertWaveLength))) for y_coord in np.arange(0,rows,1)] 


    for i in range(rows):
        
        for j in range(cols):

            offset_x = x_offsets[i] # how much to move left and right (which is based on row positon)
            offset_y = y_offsets[j] # how much to move up and down (which is based on column positon)

            # if still within the image, move the pixel value, otherwise, the output image is white anyway (255 value)
            if 0 <= i+offset_y < rows  and   0 <= j+offset_x < cols:
                img_output[i,j] = img[i+offset_y, j+offset_x]

    newBBs = []
    bbPadding = 3

    for bb in BB:
        [x1, y1, x2, y2] = bb
        
        topleftX = x1 - x_offsets[y1]
        topleftY = y1 - y_offsets[x1]

        bottomleftX = x1  - x_offsets[y2]
        bottomleftY = y2 - y_offsets[x1]

        toprightX = x2 - x_offsets[y1]
        toprightY = y1 - y_offsets[x2]

        bottomrightX = x2 - x_offsets[y2]
        bottomrightY = y2 - y_offsets[x2]

        x1 = min(topleftX, bottomleftX)  - bbPadding
        y1 = min(topleftY, toprightY)   - bbPadding

        x2 = max(toprightX, bottomrightX)   + bbPadding
        y2 = max(bottomleftY, bottomrightY)  + bbPadding
        
        newbb = [x1, y1, x2, y2]
        newBBs.append(newbb)

    return img_output, newBBs



def visualiseImage(img, BBs, showBBs = True):
    img = Image.fromarray(img)

    if showBBs:
        drawing = ImageDraw.Draw(img)
        for bb in BBs:
            drawing.rectangle(bb)

    plt.imshow(img, cmap='gray')
    plt.show()






def saveImageAndAnnotations(img, BB, CHARACTERS, TEXT, img_path, label_path, text_path, name):
    name = str(name)
    label_data = []
    for (character, bb) in zip(CHARACTERS, BB):
        label = [character, bb[0], bb[1], bb[2], bb[3]]  # format is [class label, x1, y1, x2, y2]
        label_data.append(" ".join(map(str, label)))
    #print(label_data)

    output_label_path = os.path.join(label_path, "labels_" + name + ".txt")
    with open(output_label_path, "w") as f:
        f.write("\n".join(label_data))

    TEXT = [",".join(x) for x in TEXT]
    #print(TEXT)
    output_text_path = os.path.join(text_path, "text_" + name + ".txt")
    with open(output_text_path, "w") as f:
        f.write("\n".join(TEXT))

    output_image_path = os.path.join(img_path, "img_" + name + ".png")

    cv2.imwrite(output_image_path, img)










"""

BELOW IS JUST FOR THE SINGLE LETTER IMAGES


"""


#Returns a grayscale image based on specified label of img_size
def create_letter_image(label, img_size):

    font = ImageFont.truetype('data_management/augmentation/Habbakuk.ttf',
                              random.randint(20, 35))


    #Create blank image and create a draw interface
    img = Image.new('L', img_size, 255)    
    draw = ImageDraw.Draw(img)

    translation_factor = 3
    horzontal_translation = random.randint(-translation_factor, translation_factor)
    vertical_translation = random.randint(-translation_factor, translation_factor)

    #Get size of the font and draw the token in the center of the blank image
    w,h = font.getsize(char_map[label])
    draw.text(((img_size[0]-w)/2 + horzontal_translation, (img_size[1]-h)/2+vertical_translation), char_map[label], 0, font)


    img = np.array(img) 

     
    if random.random() > 0.4:
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.erode(img,kernel,iterations = 1)  # erode the image (makes the black letters wider)
    if random.random() > 0.4:
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    kernelsize = random.randrange(3, 7+1, 2) # pick a random (odd) kernel size
    img = cv2.GaussianBlur(img, (kernelsize, kernelsize) ,0)  # softens the image, gets an "aged" look (i.e. the sharp corners/lines fade into the paper)
    
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

    return img





# generate N random images per letter in the dictionary
def HabbakukLettersGenerator(path, N=100):

    for labelIDX in range(len(list(char_map))):
        label = list(char_map)[labelIDX]
        print(label)
        print(labelIDX)

        for n in range(N):
            # make and warp an image containing just one letter
            img = create_letter_image(label, (50, 50))
            img = letterImageWarper(img)
            img = imageRotator(img)
            img = imageShearer(img)

            ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

            # make the filename, save the image (the label and label index is contained in the filename, so no need for seperate annotations)
            filename = str(label) + "_" + str(labelIDX) + "_" +  str(n) + ".png"
            output_image_path = os.path.join(path, filename)

            cv2.imwrite(output_image_path, img)