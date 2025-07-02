# Replit.md - Multimodal Image Fusion using Vision Transformers

## Overview

This project is a college-level implementation of multimodal image fusion using Vision Transformer-inspired techniques. The application processes satellite images from different modalities (VIS, IR, NIR) and combines them using attention mechanisms and multi-scale processing to create enhanced composite images for remote sensing applications.

The system features a modern web interface built with vanilla HTML, CSS, and JavaScript, backed by a Flask server that handles image processing through a custom fusion model.

## System Architecture

### Frontend Architecture
- **Technology Stack**: Vanilla HTML5, CSS3, JavaScript (ES6+)
- **Design Pattern**: Component-based vanilla JS with modular functions
- **UI Framework**: Custom CSS with CSS Grid and Flexbox layouts
- **Theming**: CSS custom properties with light/dark theme support
- **Responsive Design**: Mobile-first approach with breakpoints for tablet and desktop

### Backend Architecture
- **Framework**: Flask with minimal dependencies
- **Pattern**: MVC-inspired structure with separation of concerns
- **File Handling**: Werkzeug utilities for secure file uploads
- **Image Processing**: Custom fusion model with numpy and OpenCV
- **API Design**: RESTful endpoints with JSON responses

### Data Storage Solutions
- **File Storage**: Local filesystem with organized directory structure
  - `static/uploads/`: Temporary storage for uploaded images
  - `static/results/`: Storage for processed/fused images
- **Session Management**: Flask built-in session handling
- **No Database**: Simple file-based approach suitable for demonstration purposes

## Key Components

### 1. Web Interface (`templates/index.html`, `static/style.css`, `static/script.js`)
- **Purpose**: Provides intuitive drag-and-drop interface for image upload
- **Features**: 
  - Multi-file upload with preview
  - Real-time progress feedback
  - Theme switching capability
  - Responsive card-based layout
- **Rationale**: Clean, modern interface that showcases the technical work effectively

### 2. Flask Application (`app.py`)
- **Purpose**: HTTP server and request handling
- **Key Features**:
  - File upload validation and security
  - Image fusion orchestration
  - Error handling and logging
  - Static file serving
- **Security**: Secure filename handling, file type validation, size limits

### 3. Fusion Model (`fusion_model.py`)
- **Purpose**: Core image processing and fusion algorithms
- **Architecture**:
  - Vision Transformer-inspired patch processing
  - Multi-head attention mechanisms
  - Multi-scale feature extraction
  - PCA-based dimensionality reduction
- **Rationale**: Lightweight implementation that demonstrates advanced concepts without excessive complexity

## Data Flow

1. **Image Upload**: User selects 2-3 satellite images through web interface
2. **Client-side Validation**: JavaScript validates file types and sizes
3. **Server Upload**: Files are securely uploaded to Flask server
4. **Preprocessing**: Images are normalized, resized, and prepared for fusion
5. **Fusion Processing**: 
   - Images divided into patches (ViT-inspired)
   - Multi-scale feature extraction
   - Attention-based fusion
   - Reconstruction to final image
6. **Response**: Fused image path returned to client
7. **Display**: Client fetches and displays the result alongside inputs

## External Dependencies

### Python Dependencies
- **Flask**: Web framework for HTTP handling
- **PIL (Pillow)**: Image loading and basic processing
- **NumPy**: Numerical computations and array operations
- **OpenCV**: Advanced image processing operations
- **scikit-learn**: PCA and preprocessing utilities
- **Werkzeug**: Secure file handling utilities

### Frontend Dependencies
- **Google Fonts**: Typography (Poppins, Inter, Roboto Slab)
- **Font Awesome**: Icon library for UI elements

### Rationale for Minimal Dependencies
- **Educational Focus**: Keeps the codebase understandable for academic evaluation
- **Performance**: Lightweight stack ensures fast processing
- **Deployment**: Easier to deploy with fewer dependencies

## Deployment Strategy

### Local Development
- **Environment**: Python 3.7+ with pip-installed dependencies
- **Configuration**: Environment variables for session secrets
- **File Structure**: Static directory organization for uploads and results

### Replit Deployment
- **Auto-detection**: Flask app automatically detected by Replit
- **Static Files**: Served through Flask's static file handling
- **Persistence**: Uploaded files stored in Replit's filesystem
- **Environment**: Production-ready with proxy fix for Replit's infrastructure

### Security Considerations
- **File Validation**: Strict file type and size checking
- **Secure Filenames**: Werkzeug's secure_filename for upload safety
- **Directory Isolation**: Uploads restricted to designated directories
- **Session Security**: Configurable session secret key

## Changelog

```
Changelog:
- July 01, 2025. Initial setup
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```

---

*This document serves as a comprehensive guide for understanding and maintaining the Image Fusion application architecture. The system prioritizes educational value, clean code organization, and practical demonstration of Vision Transformer concepts in a web-based environment.*