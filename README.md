# Artify image
A tool that converts an image to the same image but the image is divided into small rectangles
and those are replaced an image of a database the replicates the color of that region the most.

![Example](https://github.com/Sfeeen/artify_image/blob/master/screenshots/done.PNG?raw=true "Title")

## Steps
* Get yourself a lot of images and click "set image database path" to select the folder where your images are stored.
* Click "convert and index images". All the images in the chosen folder will be converted to the dimensions SUB IMAGE WIDTH and SUB IMAGE HEIGHT. When artifying an image later these images might be 
scalled further down but they will always keep the same ratio. It will save all converted images in a folder and will store for each image what the average color is.
  
* You can click "Show random converted image". This will show one of your converted images with its average color and with a matrix of many small images.
Take a look from a distance and chinese-ify your eyes. You might perceive how they not differ a lot.

![Random image](https://github.com/Sfeeen/artify_image/blob/master/screenshots/show_random_image.PNG?raw=true "Title")

* click "Load main image". This will load the image that will be artified.

* "Artify main image". This will convert your image into a combination of small images from your database. It will update its image will doing so. You can opt to not reuse an image if it has already been used.
![Random image](https://github.com/Sfeeen/artify_image/blob/master/screenshots/artifying.PNG?raw=true "Title")
