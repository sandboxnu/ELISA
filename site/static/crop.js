// vars
// on change show image with crop options
const make_img = function(image) {
    // create new image
    let img = document.createElement('img');
    img.id = 'image';
    img.src = image;
    img.alt = 'Could not find image.';

    console.log('the image was found');
    // clean result before
    result.innerHTML = '';
    // append new image
    result.appendChild(img);
    // show save btn and options
    save.classList.remove('hide');
    options.classList.remove('hide');
    // init cropper
    cropper = new Cropper(img);
 }

const startup = function(img) {
    let result = document.querySelector('.result'),
    img_result = document.querySelector('.img-result'),
    img_w = document.querySelector('.img-w'),
    img_h = document.querySelector('.img-h'),
    options = document.querySelector('.options'),
    save = document.querySelector('.save'),
    cropped = document.querySelector('.cropped'),
    dwn = document.querySelector('.download'),
    upload = document.querySelector('#file-input'),
    cropper = '';

    make_img(img);

        // save on click
    save.addEventListener('click',(e)=>{
      e.preventDefault();
      // get result to data uri
      let imgSrc = cropper.getCroppedCanvas({
            width: img_w.value // input value
        }).toDataURL();
      // remove hide class of img
      cropped.classList.remove('hide');
      img_result.classList.remove('hide');
        // show image cropped
      cropped.src = imgSrc;
      dwn.classList.remove('hide');
      dwn.download = 'imagename.png';
      dwn.setAttribute('href',imgSrc);
    });

}

