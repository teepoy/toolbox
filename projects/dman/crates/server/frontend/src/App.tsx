import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import Gallery from './pages/Gallery'
import Detail from './pages/Detail'
import './App.css'

function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
      <h1 className="text-4xl font-bold text-gray-900 dark:text-white">dman</h1>
      <p className="text-lg text-gray-600 dark:text-gray-300">Dataset Manager</p>
      <p className="text-sm text-gray-400">Manage your image datasets with ease.</p>
    </div>
  )
}

function About() {
  return (
    <div className="p-8">
      <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">About</h2>
      <p className="text-gray-600 dark:text-gray-300">dman — a fast, local-first dataset manager for machine learning.</p>
    </div>
  )
}

function Nav() {
  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `px-4 py-2 rounded text-sm font-medium transition-colors ${
      isActive
        ? 'bg-indigo-600 text-white'
        : 'text-gray-600 hover:text-gray-900 dark:text-gray-300 dark:hover:text-white'
    }`

  return (
    <nav className="flex items-center gap-2 p-4 border-b border-gray-200 dark:border-gray-700">
      <span className="font-bold text-lg text-gray-900 dark:text-white mr-4">dman</span>
      <NavLink to="/" end className={linkClass}>Home</NavLink>
      <NavLink to="/datasets" className={linkClass}>Datasets</NavLink>
      <NavLink to="/about" className={linkClass}>About</NavLink>
    </nav>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-white dark:bg-gray-900">
        <Nav />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/datasets" element={<Gallery />} />
          <Route path="/datasets/:name" element={<Gallery />} />
          <Route path="/datasets/:datasetName/images/:imageId" element={<Detail />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<Home />} />
        </Routes>
      </div>
    </BrowserRouter>
  )
}
