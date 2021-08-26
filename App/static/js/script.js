var input = document.getElementById('upload');
var infoArea = document.getElementById('upload-label');


// shows selected image
function previewImage() {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        
        reader.onload = function (e) {
            $('#imageResult0')
                .attr('src', e.target.result);
        };
        reader.readAsDataURL(input.files[0]);
    }
}

$(function () {
    $('#upload').on('change', function () {
        previewImage();
        clearPrevious()
    });
});

// shows selected image name
input.addEventListener('change', showFileName);
function showFileName() {
  var fileName = input.files[0].name;
  infoArea.textContent = 'File name: ' + fileName;
}

window.onload = () => {
	$('#submit').click(() => {
		if(input.files && input.files[0])
		{
            clearPrevious()  // clear the previous results
            $('#submit').prop('disabled', true);  // disable the submit button
            $('#submit').html('Processing');  // change its text
			let formData = new FormData();
			formData.append('image' , input.files[0]);
			$.ajax({
				url: "http://localhost:8080/",
				type:"POST",
				data: formData,
				cache: false,
				processData:false,
				contentType:false,
				error: function(data){
					console.log("upload error" , data);
					console.log(data.getAllResponseHeaders());
				},
				success: function(data){
                    nr_images = data['status']  // get the number of images received  
                    // document.getElementById('container1').insertAdjacentHTML('beforeend', 
                    // `<p id="resultP" style="color: white; text-align: left; font-size: 1.5rem">${nr_images} results:</p>`)
                    for (let i=0; i<nr_images/2; i++){
                        key = 'image_' + i
                        bytestring = data[key]
                        image_original = bytestring.split('\'')[1]
                        
                        key = 'image_' + (nr_images/2+i)
                        bytestring = data[key]
                        image_final = bytestring.split('\'')[1]
                        
                        // not the cleanest thing but we move:
                        div = `<div class="imgResultContainer"><div style="display: inline-block;"><img id="imageResult${i*2+1}" src="#" alt="" class="img-fluid rounded shadow-sm mx-auto d-block"></div><div style="display: inline-block;"><img src="static/assets/arrow.png" alt="" class="img-fluid rounded shadow-sm mx-auto d-block" style="max-width: 20%;max-height: 20%"></div><div style="display: inline-block;"><img id="imageResult${i*2+2}" src="#" alt="" class="img-fluid rounded shadow-sm mx-auto d-block"></div></div>`
                        document.getElementById('container1').insertAdjacentHTML('beforeend', div)
                        $('#imageResult' + (i*2+1)).attr('src' , 'data:image/jpg;base64,' + image_original)
                        $('#imageResult' + (i*2+2)).attr('src' , 'data:image/jpg;base64,' + image_final)
                    }
                    $('#submit').prop('disabled', false);  // activating the submit button
                    $('#submit').html('Detect');   // changing its text
				}
			});
		}
	});
};

function clearPrevious(){
    $('div.imgResultContainer').remove();  // removing all the image result containers
    // $('#resultP').remove()  // and the person count message
}