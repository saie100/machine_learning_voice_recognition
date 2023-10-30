import express from 'express'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'  // Import the necessary function
const __filename = fileURLToPath(import.meta.url)  // Get the full file path
const __dirname = path.dirname(__filename)

const app = express()
const PORT = 5000
const IMAGES_DIRECTORY = path.join('images')  // Assuming images are in an 'images' subdirectory inside 'app'

// Serve static files from the current directory
app.use(express.static('app'))

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`)
})


app.get('/image-list', (req, res) => {
    const dirParam = req.query.dir

    // Ensure the directory parameter is one of the allowed values
    if (!['target_voice', 'other_voice'].includes(dirParam)) {
        return res.status(400).json({ error: 'Invalid directory' })
    }
    const dirPath = path.join(IMAGES_DIRECTORY, dirParam)
    fs.readdir(dirPath, (err, files) => {
        if (err) {
            console.error("Error reading the directory:", err)
            return res.status(500).json({ error: 'Failed to read the directory' })
        }

        // Filter out only the PNG files
        const pngFiles = files.filter(file => path.extname(file).toLowerCase() === '.png')
        res.json(pngFiles)
    })
})

app.get('/image', (req, res) => {
    const dirParam = req.query.dir
    const fileParam = req.query.file

    if (!['target_voice', 'other_voice'].includes(dirParam)) {
        return res.status(400).json({ error: 'Invalid directory' })
    }

    const dirPath = path.join(IMAGES_DIRECTORY, dirParam)
    fs.readdir(dirPath, (err, files) => {
        if (err) {
            console.error("Error reading the directory:", err)
            return res.status(500).json({ error: 'Failed to read the directory' })
        }

        // Find the requested PNG file
        const pngFile = files.find(file => file === fileParam)
        if (!pngFile) {
            console.error("Error: Image not found:", fileParam)
            return res.status(404).json({ error: 'Image not found' })
        }

        const absolutePath = path.join(dirPath, pngFile)
        res.sendFile(absolutePath, { root: __dirname })
    })
})
