import React, { useEffect, useRef } from 'react';
import icon from './image-icon.svg';
import './ImageUploader.css';

function ImageUploader() {
    const dragAreaRef = useRef(null);
    const dragTextRef = useRef(null);
    const inputRef = useRef(null);
    let file;

    const handleClick = () => {
        inputRef.current.click();
    }

    const handleChange = (event) => {
        file = event.target.files[0];
        dragAreaRef.current.classList.add('active');
        displayFile();
    }

    const handleDragOver = (event) => {
        event.preventDefault();
        dragAreaRef.current.classList.add('active');
    }

    const handleDragLeave = () => {
        dragAreaRef.current.classList.remove('active');
    }

    const handleDrop = (event) => {
        event.preventDefault();
        file = event.dataTransfer.files[0];
        displayFile();
    }

    const displayFile = () => {
        let fileType = file.type;
        let validExt = ['image/jpeg', 'image/jpg', 'image/png'];
        if (validExt.includes(fileType)) {
            let fileReader = new FileReader();

            fileReader.onload = () => {
                let fileURL = fileReader.result;
                dragAreaRef.current.innerHTML = `<img src="${fileURL}" alt="" />`;
            };
            fileReader.readAsDataURL(file);
        } else {
            alert('This file type is not supported');
            dragAreaRef.current.classList.remove('active');
        }
    }

    useEffect(() => {
        const dragArea = dragAreaRef.current;
        const dragText = dragTextRef.current;
        const input = inputRef.current;

        dragArea.addEventListener('dragover', handleDragOver);
        dragArea.addEventListener('dragleave', handleDragLeave);
        dragArea.addEventListener('drop', handleDrop);
        input.addEventListener('change', handleChange);

        return () => {
            dragArea.removeEventListener('dragover', handleDragOver);
            dragArea.removeEventListener('dragleave', handleDragLeave);
            dragArea.removeEventListener('drop', handleDrop);
            input.removeEventListener('change', handleChange);
        };
    }, []);

    return (
        <div>
            <div className="container">
                <h3> Upload your file </h3>
                <div className="drag-area" ref={dragAreaRef}>
                    <div className="icon">
                        <img src={icon} style={{ height: 50, width: 50 }} alt="image" />
                    </div>
                    <span className="header" ref={dragTextRef}> Drag & Drop </span>
                    <span className="header"> or <span className="browse-button" onClick={handleClick}> browse </span> </span>
                    <input type="file" hidden ref={inputRef} />
                    <span className="support"> Supports: JPEG, JPG, PNG </span>
                </div>
            </div>
        </div>
    );
}

export default ImageUploader;
