# PathwayLens 2.0 Frontend

Next.js frontend application for PathwayLens 2.0 - Next-generation computational biology platform.

## Features

- **Modern Tech Stack**: Next.js 14, TypeScript, Tailwind CSS
- **Design System**: Comprehensive design system with Radix UI components
- **Interactive Visualizations**: Plotly.js and D3.js integration
- **Authentication**: Secure user authentication and authorization
- **Real-time Updates**: WebSocket integration for job progress
- **Responsive Design**: Mobile-first responsive design
- **Accessibility**: WCAG 2.1 AA compliant components

## Getting Started

### Prerequisites

- Node.js 18.0.0 or later
- npm 8.0.0 or later

### Installation

1. Install dependencies:
```bash
npm install
```

2. Copy environment variables:
```bash
cp env.example .env.local
```

3. Update environment variables in `.env.local`:
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_URL=http://localhost:3000
```

4. Start the development server:
```bash
npm run dev
```

5. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint
- `npm run type-check` - Run TypeScript type checking
- `npm run test` - Run tests
- `npm run test:watch` - Run tests in watch mode
- `npm run test:coverage` - Run tests with coverage
- `npm run storybook` - Start Storybook
- `npm run build-storybook` - Build Storybook

## Project Structure

```
frontend/
├── src/
│   ├── app/                    # Next.js app router
│   │   ├── layout.tsx         # Root layout
│   │   ├── page.tsx           # Home page
│   │   ├── dashboard/         # Dashboard pages
│   │   └── auth/              # Authentication pages
│   ├── components/            # React components
│   │   ├── ui/                # Base UI components
│   │   ├── auth/              # Authentication components
│   │   ├── dashboard/         # Dashboard components
│   │   ├── layout/            # Layout components
│   │   └── sections/          # Page sections
│   ├── lib/                   # Utility libraries
│   │   ├── api/               # API client
│   │   ├── providers/         # React providers
│   │   └── utils/             # Utility functions
│   └── types/                 # TypeScript types
├── design-system/             # Design system
│   ├── tokens/                # Design tokens
│   ├── components/            # Design system components
│   └── themes/                # Theme configurations
├── public/                    # Static assets
└── docs/                      # Documentation
```

## Design System

The frontend uses a comprehensive design system built on:

- **Design Tokens**: Colors, typography, spacing, breakpoints
- **Radix UI**: Accessible, unstyled UI primitives
- **Tailwind CSS**: Utility-first CSS framework
- **Class Variance Authority**: Component variant management

### Theme Support

- Light theme (default)
- Dark theme
- High contrast theme (accessibility)

## API Integration

The frontend integrates with the PathwayLens API through:

- **Axios**: HTTP client with interceptors
- **React Query**: Data fetching and caching
- **TypeScript**: Type-safe API integration

## Authentication

- JWT-based authentication
- Protected routes
- User session management
- Role-based access control

## Visualization

Interactive visualizations powered by:

- **Plotly.js**: Scientific plotting and charts
- **D3.js**: Custom data visualizations
- **Recharts**: React charting library

## Testing

- **Jest**: Testing framework
- **React Testing Library**: Component testing
- **Storybook**: Component documentation and testing

## Deployment

The application is optimized for deployment on:

- Vercel (recommended)
- Netlify
- Docker containers
- Static hosting

## Contributing

1. Follow the existing code style
2. Write tests for new features
3. Update documentation
4. Ensure accessibility compliance

## License

MIT License - see LICENSE file for details.
