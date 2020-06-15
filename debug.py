import os
os.system('ffmpeg -r 10 -i "C:\\Users\\Kashif\\Desktop\\Flask Segmentation\\Img\\frame%01d.jpg"  -vcodec mpeg4 -y movie.mp4')