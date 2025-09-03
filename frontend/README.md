# Fashion Search Frontend

A modern, responsive Next.js web application for searching and browsing fashion products with AI-powered semantic search capabilities.

## Features

### 🎨 User Interface
- **Clean Search Interface**: Centered search bar with gradient background
- **Responsive Design**: Works seamlessly on mobile, tablet, and desktop
- **Fashion-Themed Background**: Animated floating fashion icons with subtle gradients
- **Skeleton Loading**: Smooth loading states while fetching results

### 🛍️ Product Display
- **Product Grid**: Responsive grid layout (2-6 columns based on screen size)
- **Product Cards**: Hover effects, star ratings, and pricing information
- **Image Carousel**: Full-screen product modal with multiple image navigation
- **Product Details**: Complete feature lists, specifications, and metadata

### 🔍 Search Experience
- **Mock API Integration**: Returns ranked fashion products with similarity scores
- **Search Suggestions**: Quick-access buttons for common fashion queries
- **Find Similar**: Discover products similar to any selected item
- **Real-time Results**: Instant search with skeleton loading states

### 🛠️ Developer Features
- **Debug Mode**: Toggle with Cmd/Ctrl+D to show similarity scores and rankings
- **Debug Panels**: Overlay similarity scores and ranking information
- **Keyboard Shortcuts**: Full keyboard navigation support
- **TypeScript**: Full type safety throughout the application

## Technology Stack

- **Framework**: Next.js 15 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS v4
- **UI Components**: Headless UI with Hero Icons
- **Animations**: Framer Motion
- **Build Tool**: Turbopack
- **Deployment**: Docker containerization

## Getting Started

### Prerequisites
- Node.js 18+ 
- npm or yarn
- Docker (optional, for containerization)

### Development Setup

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Start Development Server**
   ```bash
   npm run dev
   ```
   
   Open [http://localhost:3000](http://localhost:3000) in your browser.

3. **Enable Debug Mode**
   - Press `Cmd+D` (Mac) or `Ctrl+D` (Windows/Linux)
   - Or click the eye icon in the bottom-right corner
   - Shows similarity scores and ranking information

### Production Build

```bash
npm run build
npm start
```

### Docker Deployment

```bash
# Build the Docker image
docker build -t fashion-search-frontend .

# Run the container
docker run -p 3000:3000 fashion-search-frontend
```

Or use docker-compose:
```bash
docker-compose -f docker-compose.frontend.yml up --build
```

## Project Structure

```
frontend/
├── app/                    # Next.js app directory
│   ├── api/chat/          # Mock API endpoint
│   ├── globals.css        # Global styles and animations
│   ├── layout.tsx         # Root layout component
│   └── page.tsx           # Home page with search interface
├── components/            # React components
│   ├── DebugPanel.tsx     # Similarity score display
│   ├── DebugToggle.tsx    # Debug mode toggle
│   ├── ProductCard.tsx    # Individual product display
│   ├── ProductGrid.tsx    # Product grid with skeletons
│   ├── ProductModal.tsx   # Full-screen product details
│   └── SearchBar.tsx      # Search input component
├── data/                  # Mock data
│   └── amazon_fashion_sample.json
├── lib/                   # Utilities and types
│   ├── mockData.ts        # Mock API functions
│   └── types.ts           # TypeScript interfaces
├── public/                # Static assets
│   └── placeholder-product.* # Product placeholder images
├── Dockerfile             # Docker configuration
├── next.config.ts         # Next.js configuration
└── package.json           # Dependencies and scripts
```

## Key Features Demo

### Debug Mode
Press `Cmd/Ctrl+D` to toggle debug mode and see:
- Similarity scores for each product
- Ranking information
- Search strategy being used

### Mock Data
Uses 300+ real Amazon Fashion products with:
- Product images and details
- Star ratings and prices
- Features and specifications
- Realistic similarity scoring
