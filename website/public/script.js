document.addEventListener('DOMContentLoaded', () => {
    const dragArea = document.querySelector('.drag-area');
    const dragText = document.querySelector('.header');

    let button = document.querySelector('.browse-button');
    let input = document.querySelector('input[type="file"]');

    let file;

    button.onclick = () => {
        input.click();
    };

    input.addEventListener('change', function() {
        file = this.files[0];
        dragArea.classList.add('active');
        displayFile();
    });

    dragArea.addEventListener('dragover', (event) => {
        console.log('File is over dragover');

        event.preventDefault();
        dragText.textContent = 'Release to Upload';
        dragArea.classList.add('active');
    });

    dragArea.addEventListener('dragleave', () => {
        dragText.textContent = 'Drag & Drop';
        dragArea.classList.remove('active');
    });

    dragArea.addEventListener('drop', (event) => {
        event.preventDefault();
        file = event.dataTransfer.files[0];
        displayFile();
    });

    function displayFile() {
        let fileType = file.type;
        let validExt = ['image/jpeg', 'image/jpg', 'image/png'];
        if (validExt.includes(fileType)) {
            let fileReader = new FileReader();

            fileReader.onload = () => {
                let fileURL = fileReader.result;
                dragArea.innerHTML = `<img src="${fileURL}" alt="" />`;
            };
            fileReader.readAsDataURL(file);
        } else {
            alert('This file type is not supported');
            dragArea.classList.remove('active');
        }
    }
});