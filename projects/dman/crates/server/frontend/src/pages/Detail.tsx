import { useState, useEffect, useRef, useCallback } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'

interface BBox {
  x: number
  y: number
  width: number
  height: number
}

interface RawBBox {
  x: number
  y: number
  w?: number
  h?: number
  width?: number
  height?: number
  normalized?: boolean
}

interface Annotation {
  id: number
  image_id: number
  category_id: number | null
  bbox: BBox | null
  segmentation: number[][] | null
  keypoints: number[] | null
  metadata: Record<string, unknown> | null
}

interface ImageData {
  id: number
  dataset_id: number
  file_name: string
  file_path: string
  width: number | null
  height: number | null
  hash: string | null
  metadata: Record<string, unknown> | null
}

interface Category {
  id: number
  dataset_id: number
  name: string
  supercategory: string | null
}

interface ImageDetail {
  image: ImageData
  annotations: Annotation[]
}

const CATEGORY_COLORS = [
  '#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6',
  '#ec4899', '#14b8a6', '#f97316', '#06b6d4', '#84cc16',
  '#6366f1', '#a855f7', '#10b981', '#fb923c', '#38bdf8',
]

function getCategoryColor(categoryId: number | null, categoryMap: Map<number, string>): string {
  if (categoryId === null) return CATEGORY_COLORS[0]
  const keys = Array.from(categoryMap.keys()).sort((a, b) => a - b)
  const idx = keys.indexOf(categoryId)
  return CATEGORY_COLORS[(idx >= 0 ? idx : categoryId) % CATEGORY_COLORS.length]
}

function parseBBox(raw: BBox | RawBBox | null | undefined): {
  x: number; y: number; w: number; h: number; normalized: boolean
} | null {
  if (!raw) return null
  const r = raw as RawBBox
  const x = r.x
  const y = r.y
  const w = r.w ?? r.width ?? 0
  const h = r.h ?? r.height ?? 0
  const normalized = !!r.normalized
  return { x, y, w, h, normalized }
}

function bboxToPixels(
  parsed: { x: number; y: number; w: number; h: number; normalized: boolean },
  imgW: number,
  imgH: number
): { left: number; top: number; right: number; bottom: number } {
  if (parsed.normalized) {
    // YOLO: x,y are CENTER, w,h are fractions
    const left = (parsed.x - parsed.w / 2) * imgW
    const top = (parsed.y - parsed.h / 2) * imgH
    const right = (parsed.x + parsed.w / 2) * imgW
    const bottom = (parsed.y + parsed.h / 2) * imgH
    return { left, top, right, bottom }
  } else {
    // COCO / pixel: x,y are top-left
    return {
      left: parsed.x,
      top: parsed.y,
      right: parsed.x + parsed.w,
      bottom: parsed.y + parsed.h,
    }
  }
}

interface BBoxOverlayProps {
  annotations: Annotation[]
  categoryMap: Map<number, string>
  imgNaturalWidth: number
  imgNaturalHeight: number
  displayWidth: number
  displayHeight: number
}

function BBoxOverlay({
  annotations,
  categoryMap,
  imgNaturalWidth,
  imgNaturalHeight,
  displayWidth,
  displayHeight,
}: BBoxOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = displayWidth
    canvas.height = displayHeight

    ctx.clearRect(0, 0, displayWidth, displayHeight)

    if (!imgNaturalWidth || !imgNaturalHeight) return

    const scaleX = displayWidth / imgNaturalWidth
    const scaleY = displayHeight / imgNaturalHeight

    for (const ann of annotations) {
      const parsed = parseBBox(ann.bbox)
      if (!parsed) continue

      const px = bboxToPixels(parsed, imgNaturalWidth, imgNaturalHeight)
      const left = px.left * scaleX
      const top = px.top * scaleY
      const w = (px.right - px.left) * scaleX
      const h = (px.bottom - px.top) * scaleY

      const color = getCategoryColor(ann.category_id, categoryMap)
      const catName = ann.category_id !== null ? categoryMap.get(ann.category_id) : undefined

      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.strokeRect(left, top, w, h)

      ctx.fillStyle = color + '22'
      ctx.fillRect(left, top, w, h)

      if (catName) {
        const fontSize = Math.max(10, Math.min(14, Math.floor(h * 0.18)))
        ctx.font = `bold ${fontSize}px monospace`
        const textMetrics = ctx.measureText(catName)
        const labelW = textMetrics.width + 8
        const labelH = fontSize + 6

        const labelTop = top > labelH ? top - labelH : top
        ctx.fillStyle = color
        ctx.fillRect(left, labelTop, labelW, labelH)
        ctx.fillStyle = '#ffffff'
        ctx.fillText(catName, left + 4, labelTop + labelH - 4)
      }
    }
  }, [annotations, categoryMap, imgNaturalWidth, imgNaturalHeight, displayWidth, displayHeight])

  return (
    <canvas
      ref={canvasRef}
      data-testid="bbox-overlay"
      className="absolute inset-0 pointer-events-none"
      style={{ width: displayWidth, height: displayHeight }}
    />
  )
}

export default function Detail() {
  const { datasetName, imageId } = useParams<{ datasetName: string; imageId: string }>()
  const navigate = useNavigate()

  const [detail, setDetail] = useState<ImageDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [categories, setCategories] = useState<Category[]>([])
  const [categoryMap, setCategoryMap] = useState<Map<number, string>>(new Map())

  const [prevId, setPrevId] = useState<number | null>(null)
  const [nextId, setNextId] = useState<number | null>(null)

  const imgRef = useRef<HTMLImageElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [imgDimensions, setImgDimensions] = useState<{
    natural: { w: number; h: number }
    display: { w: number; h: number }
  } | null>(null)

  const numericId = imageId ? parseInt(imageId, 10) : NaN

  useEffect(() => {
    if (!datasetName || isNaN(numericId)) return
    setLoading(true)
    setError(null)

    fetch(`/api/datasets/${encodeURIComponent(datasetName)}/images/${numericId}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      .then((json: { data: ImageDetail }) => {
        setDetail(json.data)
        setLoading(false)
      })
      .catch((err) => {
        setError(err.message)
        setLoading(false)
      })
  }, [datasetName, numericId])

  useEffect(() => {
    if (!datasetName) return
    fetch(`/api/datasets/${encodeURIComponent(datasetName)}/categories`)
      .then((res) => res.ok ? res.json() : Promise.reject())
      .then((json: { data: Category[] }) => {
        setCategories(json.data)
        const map = new Map<number, string>()
        for (const cat of json.data) {
          map.set(cat.id, cat.name)
        }
        setCategoryMap(map)
      })
      .catch(() => {
        setCategories([])
        setCategoryMap(new Map())
      })
  }, [datasetName])

  useEffect(() => {
    if (!datasetName || !detail) return

    const datasetId = detail.image.dataset_id

    fetch(
      `/api/datasets/${encodeURIComponent(datasetName)}/images?per_page=500&page=1`
    )
      .then((res) => res.ok ? res.json() : Promise.reject())
      .then((json: { data: Array<{ id: number }>; pagination: { total: number } }) => {
        const ids = json.data.map((img) => img.id).sort((a, b) => a - b)
        const idx = ids.indexOf(numericId)
        setPrevId(idx > 0 ? ids[idx - 1] : null)
        setNextId(idx >= 0 && idx < ids.length - 1 ? ids[idx + 1] : null)

        if (json.pagination.total > 500) {
          const perPage = 500
          const pages = Math.ceil(json.pagination.total / perPage)
          const pagePromises: Promise<{ data: Array<{ id: number }> }>[] = []
          for (let p = 2; p <= pages; p++) {
            pagePromises.push(
              fetch(
                `/api/datasets/${encodeURIComponent(datasetName)}/images?per_page=500&page=${p}`
              ).then((r) => r.ok ? r.json() : Promise.reject())
            )
          }
          Promise.all(pagePromises).then((pages) => {
            const allIds = ids.slice()
            for (const page of pages) {
              for (const img of page.data) {
                allIds.push(img.id)
              }
            }
            allIds.sort((a, b) => a - b)
            const newIdx = allIds.indexOf(numericId)
            setPrevId(newIdx > 0 ? allIds[newIdx - 1] : null)
            setNextId(newIdx >= 0 && newIdx < allIds.length - 1 ? allIds[newIdx + 1] : null)
          }).catch(() => {})
        }
      })
      .catch(() => {})

    void datasetId
  }, [datasetName, detail, numericId])

  const navigateToPrev = useCallback(() => {
    if (prevId !== null && datasetName) {
      navigate(`/datasets/${datasetName}/images/${prevId}`)
    }
  }, [prevId, datasetName, navigate])

  const navigateToNext = useCallback(() => {
    if (nextId !== null && datasetName) {
      navigate(`/datasets/${datasetName}/images/${nextId}`)
    }
  }, [nextId, datasetName, navigate])

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'ArrowLeft') navigateToPrev()
      else if (e.key === 'ArrowRight') navigateToNext()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [navigateToPrev, navigateToNext])

  function handleImageLoad() {
    const img = imgRef.current
    if (!img) return
    setImgDimensions({
      natural: { w: img.naturalWidth, h: img.naturalHeight },
      display: { w: img.offsetWidth, h: img.offsetHeight },
    })
  }

  useEffect(() => {
    const img = imgRef.current
    if (!img) return
    const observer = new ResizeObserver(() => {
      if (img.naturalWidth && img.naturalHeight) {
        setImgDimensions({
          natural: { w: img.naturalWidth, h: img.naturalHeight },
          display: { w: img.offsetWidth, h: img.offsetHeight },
        })
      }
    })
    observer.observe(img)
    return () => observer.disconnect()
  }, [detail])

  if (loading) {
    return (
      <div className="p-8 flex items-center justify-center min-h-[60vh]">
        <p className="text-gray-500 dark:text-gray-400">Loading image…</p>
      </div>
    )
  }

  if (error || !detail) {
    return (
      <div className="p-8">
        <p className="text-red-600 dark:text-red-400">
          {error ? `Error: ${error}` : 'Image not found.'}
        </p>
        <Link
          to={`/datasets/${datasetName}`}
          className="mt-4 inline-block text-indigo-600 dark:text-indigo-400 hover:underline"
        >
          ← Back to gallery
        </Link>
      </div>
    )
  }

  const { image, annotations } = detail
  const imageUrl = `/images/${image.dataset_id}/${encodeURIComponent(image.file_name)}`
  const annotationsWithBBox = annotations.filter((a) => a.bbox !== null)

  return (
    <div className="flex flex-col lg:flex-row h-[calc(100vh-57px)] overflow-hidden">
      <div className="flex-1 flex flex-col bg-gray-950 dark:bg-gray-950 overflow-hidden">
        <div className="flex items-center gap-3 px-4 py-2 bg-gray-900/80 border-b border-gray-800">
          <Link
            to={`/datasets/${datasetName}`}
            className="text-gray-400 hover:text-white transition-colors text-sm"
          >
            ← Gallery
          </Link>
          <span className="text-gray-600 text-sm">/</span>
          <span className="text-gray-300 text-sm truncate max-w-xs">{image.file_name}</span>
          <div className="ml-auto flex items-center gap-2">
            <button
              onClick={navigateToPrev}
              disabled={prevId === null}
              className="px-3 py-1 rounded text-sm bg-gray-800 text-gray-300 hover:bg-gray-700 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
              title="Previous image (←)"
            >
              ← Prev
            </button>
            <button
              onClick={navigateToNext}
              disabled={nextId === null}
              className="px-3 py-1 rounded text-sm bg-gray-800 text-gray-300 hover:bg-gray-700 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
              title="Next image (→)"
            >
              Next →
            </button>
          </div>
        </div>

        <div className="flex-1 flex items-center justify-center overflow-auto p-4">
          <div
            ref={containerRef}
            className="relative inline-block"
            style={{ maxWidth: '100%', maxHeight: '100%' }}
          >
            <img
              ref={imgRef}
              data-testid="detail-image"
              src={imageUrl}
              alt={image.file_name}
              className="block max-w-full max-h-[calc(100vh-120px)] object-contain"
              onLoad={handleImageLoad}
            />
            {imgDimensions && annotationsWithBBox.length > 0 && (
              <BBoxOverlay
                annotations={annotations}
                categoryMap={categoryMap}
                imgNaturalWidth={imgDimensions.natural.w}
                imgNaturalHeight={imgDimensions.natural.h}
                displayWidth={imgDimensions.display.w}
                displayHeight={imgDimensions.display.h}
              />
            )}
            {imgDimensions && annotationsWithBBox.length === 0 && (
              <canvas
                data-testid="bbox-overlay"
                className="absolute inset-0 pointer-events-none"
                style={{ width: imgDimensions.display.w, height: imgDimensions.display.h }}
              />
            )}
          </div>
        </div>
      </div>

      <div className="w-full lg:w-80 xl:w-96 bg-white dark:bg-gray-900 border-l border-gray-200 dark:border-gray-700 overflow-y-auto flex-shrink-0">
        <div className="p-4 space-y-4">
          <section>
            <h2 className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-2">
              Image
            </h2>
            <div className="space-y-1">
              <MetaRow label="ID" value={String(image.id)} />
              <MetaRow label="Filename" value={image.file_name} />
              <MetaRow label="Dataset" value={datasetName ?? ''} />
              <MetaRow label="Dataset ID" value={String(image.dataset_id)} />
              {image.width && image.height && (
                <MetaRow label="Size" value={`${image.width} × ${image.height}`} />
              )}
              {image.hash && (
                <MetaRow label="Hash" value={image.hash.slice(0, 16) + '…'} mono />
              )}
              <MetaRow label="Annotations" value={String(annotations.length)} />
            </div>
          </section>

          {image.metadata && Object.keys(image.metadata).length > 0 && (
            <section>
              <h2 className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-2">
                Metadata
              </h2>
              <pre className="text-xs bg-gray-50 dark:bg-gray-800 rounded p-2 overflow-x-auto text-gray-700 dark:text-gray-300">
                {JSON.stringify(image.metadata, null, 2)}
              </pre>
            </section>
          )}

          {annotations.length > 0 && (
            <section>
              <h2 className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-2">
                Annotations ({annotations.length})
              </h2>
              <div className="space-y-2">
                {annotations.map((ann) => {
                  const catName = ann.category_id !== null
                    ? categoryMap.get(ann.category_id)
                    : undefined
                  const color = getCategoryColor(ann.category_id, categoryMap)
                  return (
                    <div
                      key={ann.id}
                      className="rounded border border-gray-200 dark:border-gray-700 p-2 text-xs"
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <div
                          className="w-3 h-3 rounded-sm flex-shrink-0"
                          style={{ backgroundColor: color }}
                        />
                        <span className="font-medium text-gray-800 dark:text-gray-200">
                          {catName ?? `cat #${ann.category_id}`}
                        </span>
                        <span className="ml-auto text-gray-400">#{ann.id}</span>
                      </div>
                      {ann.bbox && (
                        <div className="text-gray-500 dark:text-gray-400 font-mono">
                          bbox: x={ann.bbox.x.toFixed(4)} y={ann.bbox.y.toFixed(4)}{' '}
                          w={ann.bbox.width.toFixed(4)} h={ann.bbox.height.toFixed(4)}
                        </div>
                      )}
                      {ann.metadata && Object.keys(ann.metadata).length > 0 && (
                        <details className="mt-1">
                          <summary className="cursor-pointer text-gray-400 hover:text-gray-600">
                            metadata
                          </summary>
                          <pre className="mt-1 bg-gray-50 dark:bg-gray-800 rounded p-1 overflow-x-auto text-gray-600 dark:text-gray-300">
                            {JSON.stringify(ann.metadata, null, 2)}
                          </pre>
                        </details>
                      )}
                    </div>
                  )
                })}
              </div>
            </section>
          )}

          {annotations.length === 0 && (
            <section>
              <h2 className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-2">
                Annotations
              </h2>
              <p className="text-xs text-gray-400 dark:text-gray-500">No annotations</p>
            </section>
          )}

          {categories.length > 0 && (
            <section>
              <h2 className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-2">
                Categories
              </h2>
              <div className="flex flex-wrap gap-1">
                {categories.map((cat) => (
                  <span
                    key={cat.id}
                    className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs text-white"
                    style={{ backgroundColor: getCategoryColor(cat.id, categoryMap) }}
                  >
                    {cat.name}
                  </span>
                ))}
              </div>
            </section>
          )}
        </div>
      </div>
    </div>
  )
}

function MetaRow({
  label,
  value,
  mono,
}: {
  label: string
  value: string
  mono?: boolean
}) {
  return (
    <div className="flex items-start gap-2 text-xs">
      <span className="text-gray-500 dark:text-gray-400 min-w-[80px] flex-shrink-0">{label}</span>
      <span
        className={`text-gray-800 dark:text-gray-200 break-all ${
          mono ? 'font-mono' : ''
        }`}
      >
        {value}
      </span>
    </div>
  )
}
