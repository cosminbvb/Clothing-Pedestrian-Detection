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
				url: "http://127.0.0.1:5000/",
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
                    document.getElementById('container1').insertAdjacentHTML('beforeend', 
                    '<p id="peopleCount" style="color: white; text-align: center; font-size: 1.5rem">' + nr_images/2 + ' people detected:</p>')
                    // nr_images / 2 because each person has a raw and modified image
                    for (let i=0; i<nr_images; i++){
                        key = 'image_' + i
                        bytestring = data[key]
                        image = bytestring.split('\'')[1]
                        //hardcode below:
                        //change asap
                        div = '<div class="image-area mt-4 imgResultContainer"><img id="imageResult" src="#" alt="" class="img-fluid rounded shadow-sm mx-auto d-block"></div>'
                        div = div.replace("imageResult", "imageResult" + (i+1))
                        document.getElementById('container1').insertAdjacentHTML('beforeend', div)
                        $('#imageResult' + (i+1)).attr('src' , 'data:image/jpg;base64,'+image)
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
    $('#peopleCount').remove()  // and the person count message
}