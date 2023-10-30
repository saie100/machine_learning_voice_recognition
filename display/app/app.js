// JavaScript to dynamically add images to the gallery
const galleryElement = document.getElementById('imageGallery')
const modal = document.getElementById('imageModal')
const modalImage = document.getElementById('modalImage')

function openModal(src) {
    modalImage.src = src
    modal.style.display = 'block'

}

// Close the modal when clicking outside the image
modal.addEventListener('click', (event) => {
    if (event.target === modal) {
        modal.style.display = 'none'
    }
})



function reloadImages(directory, files) {
    files.forEach(file => {
        const path = `/image/?dir=${directory}&file=${file}`
        const imgElement = document.createElement('img')
        const dirFile = `${directory}/${file}`
        imgElement.src = path
        imgElement.id = 'path'
        imgElement.alt = dirFile
        imgElement.title = dirFile
        imgElement.addEventListener('click', () => {
            openModal(imgElement.src)
        })
        const descElement = document.createElement('div')
        descElement.textContent = dirFile
        descElement.classList.add('image-description')

        const container = document.createElement('div')
        container.classList.add('image-container')
        container.appendChild(imgElement)
        container.appendChild(descElement)

        galleryElement.appendChild(container)
    })

}


function loadImagesFromDirectory(directory) {
    fetch(`/image-list?dir=${directory}`)
        .then(response => response.json())
        .then(files => {
            galleryElement.innerHTML = ''
            reloadImages(directory, files)
        })
        .catch(error => {
            console.error("Error fetching images:", error)
        })
}

document.getElementById('linkTargetVoice').addEventListener('click', function (e) {
    e.preventDefault()
    loadImagesFromDirectory('target_voice')
})

document.getElementById('linkOtherVoice').addEventListener('click', function (e) {
    e.preventDefault()
    loadImagesFromDirectory('other_voice')
})