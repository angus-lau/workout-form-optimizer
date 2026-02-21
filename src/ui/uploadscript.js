document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const uploadBtn = document.getElementById('uploadBtn');
    const progressBar = document.getElementById('progressBar');
    const statusText = document.getElementById('statusText');
    const form = document.getElementById('uploadForm');
    const errorMessage = document.getElementById('errorMessage');
    const removeBtn = document.getElementById('removeBtn');

    const selectBtns = document.querySelectorAll('.select-btn');
    const exerciseInput = document.getElementById('exerciseTypeInput');

    // for each button in bar, add event listener for click
    // when clicked, remove active class from all buttons (active = pressed down style) and add active class to clicked button
    // reads the clicked button's data-value attribute and updates the hidden input value with it
    selectBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            selectBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            exerciseInput.value = btn.getAttribute('data-value');
        });
    });


    // Error handling helpers
    function showError(text) {
        errorMessage.textContent = text;
        errorMessage.style.display = 'block';
        uploadBtn.disabled = true;
        fileInfo.style.display = 'none'; // Hide info if there is an error
    }

    function clearError() {
        errorMessage.textContent = "";
        errorMessage.style.display = 'none';
    }

    // Check if file is .mp4 and < 100 MB
    function isValidFile(file) {
        if (!(file.type === 'video/mp4') || !file.name.toLowerCase().endsWith('.mp4')) {
            showError("Only .mp4 videos are allowed");
            return false;
        } else if (file.size > 100 * 1024 * 1024) { // 100 MB limit
            showError("File size exceeds 100 MB");
            return false;
        }
        return true;
    }

    // Update the UI with file info
    function updateUI(file) {
        clearError();
        fileInfo.style.display = 'block';
        fileName.textContent = file.name;

        removeBtn.style.display = 'inline-block';
        removeBtn.addEventListener('click', () => {
            fileInput.value = '';
            fileInfo.style.display = 'none';
            uploadBtn.disabled = true;
            removeBtn.style.display = 'none';
        });

        // Convert bytes to MB
        fileSize.textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB'; 
        
        uploadBtn.disabled = false;
        statusText.textContent = "Ready to analyze";
        statusText.style.color = "#4A90E2";
        progressBar.style.width = '0%';
    }

    
    
    // Adds eventListener for clicking on the drop zone. When clicked, it triggers the hidden file input click.
    dropZone.addEventListener('click', () => fileInput.click());

    // plays dragover animation when file is dragged over the drop zone. It prevents the default behavior (automatic
    // file opening to allow dropping.
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    // Remove animation on drag leave
    ['dragleave', 'dragend'].forEach(type => {
        dropZone.addEventListener(type, () => dropZone.classList.remove('dragover'));
    });

    // Handle File Drop
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        
        if (e.dataTransfer.files.length == 1) {
            const file = e.dataTransfer.files[0];
            if (isValidFile(file)) {
                fileInput.files = e.dataTransfer.files; // Sync with hidden input
                updateUI(file);
            }
        } else {
            showError("Please drop only one file at a time");
        }
    });

    // Handle "Browse" selection
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length == 1) {
            const file = fileInput.files[0];
            if (isValidFile(file)) {
                updateUI(file);
            } else {
                showError("Only .mp4 videos are allowed");
                fileInput.value = ''; // Clear the invalid input
            }
        } else {
            showError("Please drop only one file at a time");
        }
    });

    // --- 4. Simulate Submission ---
    form.addEventListener('submit', (e) => {
        e.preventDefault(); 
        
        // uploadBtn.disabled = true;
        // statusText.textContent = `Uploading ${exerciseInput.value} video...`;
        
        // // Simulate progress bar
        // let progress = 0;
        // const interval = setInterval(() => {
        //     progress += 5;
        //     progressBar.style.width = progress + '%';
            
        //     if (progress >= 100) {
        //         clearInterval(interval);
        //         statusText.textContent = "Analysis Complete! ✅";
        //         statusText.style.color = "green";
        //         uploadBtn.disabled = false;
                
        //         alert(`SUCCESS!\n\nVideo: ${fileName.textContent}\nExercise: ${exerciseInput.value}`);
        //     }
        // }, 100);


    });
});