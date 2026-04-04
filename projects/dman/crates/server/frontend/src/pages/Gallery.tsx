import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'

interface Dataset {
  id: number
  name: string
  path: string
  format: string
}

interface Image {
  id: number
  dataset_id: number
  file_name: string
  file_path: string
  width: number | null
  height: number | null
  hash: string | null
  metadata: Record<string, unknown> | null
}

interface Pagination {
  page: number
  per_page: number
  total: number
}

interface Category {
  id: number
  dataset_id: number
  name: string
  supercategory: string | null
}

export default function Gallery() {
  const { name: routeName } = useParams<{ name?: string }>()
  const navigate = useNavigate()

  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [datasetsLoading, setDatasetsLoading] = useState(true)
  const [datasetsError, setDatasetsError] = useState<string | null>(null)

  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null)

  const [categories, setCategories] = useState<Category[]>([])
  const [categoryFilter, setCategoryFilter] = useState<string>('')

  const [images, setImages] = useState<Image[]>([])
  const [pagination, setPagination] = useState<Pagination | null>(null)
  const [page, setPage] = useState(1)
  const [imagesLoading, setImagesLoading] = useState(false)
  const [imagesError, setImagesError] = useState<string | null>(null)

  const PAGE_SIZE = 50

  useEffect(() => {
    setDatasetsLoading(true)
    setDatasetsError(null)

    fetch('/api/datasets?per_page=500')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      .then((json: { data: Dataset[] }) => {
        setDatasets(json.data)
        setDatasetsLoading(false)
      })
      .catch((err) => {
        setDatasetsError(err.message)
        setDatasetsLoading(false)
      })
  }, [])

  useEffect(() => {
    if (datasets.length === 0) return

    if (routeName) {
      const found = datasets.find((d) => d.name === routeName)
      if (found) {
        setSelectedDataset(found)
      } else {
        setSelectedDataset(datasets[0])
        navigate(`/datasets/${datasets[0].name}`, { replace: true })
      }
    } else {
      setSelectedDataset(datasets[0])
      navigate(`/datasets/${datasets[0].name}`, { replace: true })
    }
  }, [datasets, routeName]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!selectedDataset) return

    setCategories([])
    setCategoryFilter('')

    fetch(`/api/datasets/${encodeURIComponent(selectedDataset.name)}/categories`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      .then((json: { data: Category[] }) => {
        setCategories(json.data)
      })
      .catch(() => {
        setCategories([])
      })
  }, [selectedDataset])

  useEffect(() => {
    if (!selectedDataset) return

    setImagesLoading(true)
    setImagesError(null)

    const params = new URLSearchParams({
      page: String(page),
      per_page: String(PAGE_SIZE),
    })
    if (categoryFilter) {
      params.set('category', categoryFilter)
    }

    fetch(
      `/api/datasets/${encodeURIComponent(selectedDataset.name)}/images?${params.toString()}`
    )
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      .then((json: { data: Image[]; pagination: Pagination }) => {
        setImages(json.data)
        setPagination(json.pagination)
        setImagesLoading(false)
      })
      .catch((err) => {
        setImagesError(err.message)
        setImagesLoading(false)
      })
  }, [selectedDataset, page, categoryFilter])

  function handleDatasetChange(e: React.ChangeEvent<HTMLSelectElement>) {
    const ds = datasets.find((d) => d.name === e.target.value)
    if (!ds) return
    setSelectedDataset(ds)
    setPage(1)
    setCategoryFilter('')
    navigate(`/datasets/${ds.name}`)
  }

  function handleCategoryChange(e: React.ChangeEvent<HTMLSelectElement>) {
    setCategoryFilter(e.target.value)
    setPage(1)
  }

  const totalPages = pagination ? Math.max(1, Math.ceil(pagination.total / PAGE_SIZE)) : 1
  const hasPrev = page > 1
  const hasNext = page < totalPages

  if (datasetsLoading) {
    return (
      <div className="p-8 flex items-center justify-center min-h-[60vh]">
        <p className="text-gray-500 dark:text-gray-400">Loading datasets…</p>
      </div>
    )
  }

  if (datasetsError) {
    return (
      <div className="p-8">
        <p className="text-red-600 dark:text-red-400">
          Failed to load datasets: {datasetsError}
        </p>
      </div>
    )
  }

  if (datasets.length === 0) {
    return (
      <div className="p-8 flex flex-col items-center justify-center min-h-[60vh] gap-2">
        <p className="text-gray-700 dark:text-gray-300 text-lg font-medium">No datasets found</p>
        <p className="text-gray-400 text-sm">
          Import a dataset first using the <code>dman import</code> command.
        </p>
      </div>
    )
  }

  return (
    <div className="p-6 space-y-4">
      {/* ── Controls bar ── */}
      <div className="flex flex-wrap gap-4 items-center">
        <div className="flex items-center gap-2">
          <label
            htmlFor="dataset-selector"
            className="text-sm font-medium text-gray-700 dark:text-gray-300"
          >
            Dataset
          </label>
          <select
            id="dataset-selector"
            data-testid="dataset-selector"
            value={selectedDataset?.name ?? ''}
            onChange={handleDatasetChange}
            className="rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-white px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            {datasets.map((ds) => (
              <option key={ds.id} value={ds.name}>
                {ds.name}
              </option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-2">
          <label
            htmlFor="category-filter"
            className="text-sm font-medium text-gray-700 dark:text-gray-300"
          >
            Category
          </label>
          <select
            id="category-filter"
            data-testid="category-filter"
            value={categoryFilter}
            onChange={handleCategoryChange}
            className="rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-white px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            <option value="">All categories</option>
            {categories.map((cat) => (
              <option key={cat.id} value={cat.name}>
                {cat.name}
              </option>
            ))}
          </select>
        </div>

        {pagination && (
          <span className="ml-auto text-sm text-gray-500 dark:text-gray-400">
            {pagination.total === 0
              ? 'No images'
              : `${(page - 1) * PAGE_SIZE + 1}–${Math.min(page * PAGE_SIZE, pagination.total)} of ${pagination.total}`}
          </span>
        )}
      </div>

      {imagesLoading ? (
        <div className="flex items-center justify-center py-24">
          <p className="text-gray-500 dark:text-gray-400">Loading images…</p>
        </div>
      ) : imagesError ? (
        <div className="py-12 text-center">
          <p className="text-red-600 dark:text-red-400">Error loading images: {imagesError}</p>
        </div>
      ) : images.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 gap-2">
          <p className="text-gray-700 dark:text-gray-300 text-lg font-medium">No images found</p>
          {categoryFilter && (
            <p className="text-gray-400 text-sm">
              No images in category "{categoryFilter}". Try another filter.
            </p>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
          {images.map((img) => (
            <div
              key={img.id}
              className="group relative aspect-square overflow-hidden rounded-lg bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 hover:border-indigo-400 dark:hover:border-indigo-500 transition-colors"
            >
              <img
                data-testid="gallery-image"
                src={`/images/${img.dataset_id}/${encodeURIComponent(img.file_name)}`}
                alt={img.file_name}
                className="w-full h-full object-cover"
                loading="lazy"
                onError={(e) => {
                  const target = e.currentTarget
                  target.style.display = 'none'
                  const parent = target.parentElement
                  if (parent && !parent.querySelector('.img-error')) {
                    const err = document.createElement('div')
                    err.className =
                      'img-error absolute inset-0 flex items-center justify-center text-xs text-gray-400 p-2 text-center'
                    err.textContent = img.file_name
                    parent.appendChild(err)
                  }
                }}
              />
              <div className="absolute bottom-0 left-0 right-0 bg-black/60 text-white text-xs px-2 py-1 opacity-0 group-hover:opacity-100 transition-opacity truncate">
                {img.file_name}
              </div>
            </div>
          ))}
        </div>
      )}

      {pagination && pagination.total > PAGE_SIZE && (
        <div className="flex items-center justify-center gap-3 pt-2">
          <button
            onClick={() => setPage((p) => p - 1)}
            disabled={!hasPrev}
            className="px-4 py-2 rounded border border-gray-300 dark:border-gray-600 text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
          >
            ← Prev
          </button>
          <span className="text-sm text-gray-600 dark:text-gray-400">
            Page {page} / {totalPages}
          </span>
          <button
            onClick={() => setPage((p) => p + 1)}
            disabled={!hasNext}
            className="px-4 py-2 rounded border border-gray-300 dark:border-gray-600 text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
          >
            Next →
          </button>
        </div>
      )}
    </div>
  )
}
