# -*- coding: cp1251 -*-
import telebot
from telebot import types
import config
import cv2
import math
import numpy as np
from PIL import Image
import io
import torch
from torch import nn
from torch.nn import functional as F

src = ''
in_work = False

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=9, padding=2, padding_mode='replicate') # padding mode same as original Caffe code
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, padding=2, padding_mode='replicate')
        self.conv3 = nn.Conv2d(64, 1, kernel_size=5, padding=2, padding_mode='replicate')
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    
def u(s, a):
    
    if (abs(s) >= 0) & (abs(s) <= 1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
        
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    return 0

def padding(img, H, W, C):
    zimg = np.zeros((H+4, W+4, C))
    zimg[2:H+2, 2:W+2, :C] = img
      
    zimg[2:H+2, 0:2, :C] = img[:, 0:1, :C]
    zimg[H+2:H+4, 2:W+2, :] = img[H-1:H, :, :]
    zimg[2:H+2, W+2:W+4, :] = img[:, W-1:W, :]
    zimg[0:2, 2:W+2, :C] = img[0:1, :, :C]
      
    zimg[0:2, 0:2, :C] = img[0, 0, :C]
    zimg[H+2:H+4, 0:2, :C] = img[H-1, 0, :C]
    zimg[H+2:H+4, W+2:W+4, :C] = img[H-1, W-1, :C]
    zimg[0:2, W+2:W+4, :C] = img[0, W-1, :C]
      
    return zimg

def bicubic(img, ratio, a):

    H, W, C = img.shape
    img = padding(img, H, W, C)
    dH = math.floor(H*ratio)
    dW = math.floor(W*ratio)
    dst = np.zeros((dH, dW, 3))

    h = 1/ratio

    inc = 0

    for c in range(C):
        for j in range(dH):
            for i in range(dW):
                
                x, y = i * h + 2, j * h + 2

                x1 = 1 + x - math.floor(x)
                x2 = x - math.floor(x)
                x3 = math.floor(x) + 1 - x
                x4 = math.floor(x) + 2 - x

                y1 = 1 + y - math.floor(y)
                y2 = y - math.floor(y)
                y3 = math.floor(y) + 1 - y
                y4 = math.floor(y) + 2 - y

                mat_l = np.matrix([[u(x1, a), u(x2, a), u(x3, a), u(x4, a)]])
                mat_m = np.matrix([[img[int(y-y1), int(x-x1), c],
                                    img[int(y-y2), int(x-x1), c],
                                    img[int(y+y3), int(x-x1), c],
                                    img[int(y+y4), int(x-x1), c]],
                                   [img[int(y-y1), int(x-x2), c],
                                    img[int(y-y2), int(x-x2), c],
                                    img[int(y+y3), int(x-x2), c],
                                    img[int(y+y4), int(x-x2), c]],
                                   [img[int(y-y1), int(x+x3), c],
                                    img[int(y-y2), int(x+x3), c],
                                    img[int(y+y3), int(x+x3), c],
                                    img[int(y+y4), int(x+x3), c]],
                                   [img[int(y-y1), int(x+x4), c],
                                    img[int(y-y2), int(x+x4), c],
                                    img[int(y+y3), int(x+x4), c],
                                    img[int(y+y4), int(x+x4), c]]])
                mat_r = np.matrix(
                    [[u(y1, a)], [u(y2, a)], [u(y3, a)], [u(y4, a)]])

                dst[j, i, c] = np.dot(np.dot(mat_l, mat_m), mat_r)
    return dst

def result_photo(src_0, scale = 1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SRCNN().to(device)
    model.load_state_dict(torch.load('srcnn.pth'))
    outp = []
    image = cv2.imread(src_0, cv2.IMREAD_COLOR)
    if scale != 1:
        image = bicubic(image, scale, -1/2)
    for i in range(3):
        im = image[:,:,i]
        im = im.reshape(im.shape[0], im.shape[1], 1)
        im = im / 255.
        model.eval()
        with torch.no_grad():
            im = np.transpose(im, (2, 0, 1)).astype(np.float32)
            im = torch.tensor(im, dtype=torch.float).to(device)
            im = im.unsqueeze(0)
            outputs = model(im)
            outputs = outputs.cpu()
            outputs = outputs.detach().numpy()
            outputs = outputs.reshape(outputs.shape[2], outputs.shape[3], outputs.shape[1])
            outp.append(outputs)
    res = np.concatenate((outp[0], outp[1], outp[2]), axis=2)
    src = "tmp/result.png";
    cv2.imwrite(src, res*255)  

bot  = telebot.TeleBot(config.TOKEN)

@bot.message_handler()
def send_info(message):
    if in_work:
        bot.send_message(message.chat.id, "Работаю...")
    else:
        bot.send_message(message.chat.id, "Пришлите фото для улучшения.")

@bot.message_handler(content_types=['document'])
def handle_docs_photo(message):
    global in_work
    global src
    try:
        in_work = True
        chat_id = message.chat.id
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        src = "tmp/" + message.document.file_name;
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)
        rmk = types.ReplyKeyboardMarkup()
        rmk.add(types.KeyboardButton("1"), types.KeyboardButton("2"), types.KeyboardButton("4"))
        
        msg = bot.send_message(message.chat.id, "Во сколько раз хотите увеличить изображение?", reply_markup = rmk)
        bot.register_next_step_handler(msg, scale_image)
        '''result_photo(src)
        bot.send_photo(message.chat.id, open("tmp/result.png", 'rb'))  '''      
    except Exception as e:
        in_work = False
        bot.reply_to(message, e)
        
def scale_image(message):
    global in_work
    remove_k = types.ReplyKeyboardRemove(selective=False)
    try:
        bot.send_message(message.chat.id, "Обрабатываю...", reply_markup=remove_k)
        if message.text == "1":
            result_photo(src, 1)
        elif message.text == "2":
            result_photo(src, 2)
        elif message.text == "4":
            result_photo(src, 4)
        bot.send_photo(message.chat.id, open("tmp/result.png", 'rb'))
        in_work = False
    except Exception as e:
        bot.reply_to(message, e, reply_markup=remove_k)
        in_work = False
        
bot.polling()