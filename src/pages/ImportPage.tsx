import { useCallback, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import {
  UploadCloud, X, Check, Trash2, Loader2, ImagePlus, Save, Info,
  AlertCircle, Crop, Eye,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from '@/components/ui/select'
import { api, type DetectionDraft, type PhotoResult, DATE_SOURCE_LABEL } from '@/lib/api'
import { useMeta } from '@/hooks/useMeta'

interface DraftEdit extends DetectionDraft {
  checked: boolean
}

interface PhotoEntry {
  result: PhotoResult
  drafts: DraftEdit[]
  saved: boolean
}

const API_BASE = ''

function addDaysStr(dateStr: string, days: number) {
  const d = new Date(dateStr)
  d.setDate(d.getDate() + days)
  return d.toISOString().slice(0, 10)
}

export default function ImportPage() {
  const navigate = useNavigate()
  const { categories } = useMeta()
  const [photos, setPhotos] = useState<PhotoEntry[]>([])
  const [activeIdx, setActiveIdx] = useState(0)
  const [recognizing, setRecognizing] = useState(false)
  const [saving, setSaving] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const active = photos[activeIdx]

  // ---------- 上传识别 ----------
  const handleFiles = useCallback(async (files: File[]) => {
    const images = files.filter(f => f.type.startsWith('image/'))
    if (images.length === 0) {
      toast.error('请选择图片文件（jpg/png/webp）')
      return
    }
    setRecognizing(true)
    const tid = toast.loading(`正在识别 ${images.length} 张照片（YOLO 检测 + OCR 文字 + CLIP 视觉）…`)
    try {
      const results = await api.recognize(images)
      const entries: PhotoEntry[] = results.map(r => ({
        result: r,
        drafts: r.drafts.map(d => ({ ...d, checked: true })),
        saved: false,
      }))
      setPhotos(prev => [...prev, ...entries])
      if (photos.length === 0) setActiveIdx(0)
      const total = results.reduce((s, r) => s + r.drafts.length, 0)
      toast.success(`识别完成：${results.length} 张照片，共 ${total} 条待确认草稿`, { id: tid })
    } catch (e) {
      toast.error(`识别失败：${(e as Error).message}`, { id: tid })
    } finally {
      setRecognizing(false)
    }
  }, [photos.length])

  // ---------- 草稿编辑 ----------
  const patchDraft = (pi: number, di: number, patch: Partial<DraftEdit>) => {
    setPhotos(prev => prev.map((p, i) => i !== pi ? p : {
      ...p,
      drafts: p.drafts.map((d, j) => j !== di ? d : { ...d, ...patch }),
    }))
  }

  const recomputeExpiry = (d: DraftEdit, patch: Partial<DraftEdit>): Partial<DraftEdit> => {
    const prod = patch.production_date !== undefined ? patch.production_date : d.production_date
    const days = patch.shelf_life_days !== undefined ? patch.shelf_life_days : d.shelf_life_days
    if (prod && days && days > 0) return { ...patch, expiry_date: addDaysStr(prod, days) }
    return patch
  }

  const applyCandidate = (pi: number, di: number, name: string, categoryId: number,
                          categoryName: string, defaultDays: number) => {
    const d = photos[pi].drafts[di]
    const hasDate = d.date_source === 'ocr'
    let patch: Partial<DraftEdit> = { name, category_id: categoryId, category_name: categoryName }
    if (!hasDate && !d.shelf_life_days) {
      patch = recomputeExpiry(d, { ...patch, shelf_life_days: defaultDays, date_source: 'category_default' })
    }
    patchDraft(pi, di, patch)
  }

  const removeDraft = (pi: number, di: number) => {
    setPhotos(prev => prev.map((p, i) => i !== pi ? p : {
      ...p, drafts: p.drafts.filter((_, j) => j !== di),
    }))
  }

  const removePhoto = (pi: number) => {
    setPhotos(prev => prev.filter((_, i) => i !== pi))
    setActiveIdx(idx => (idx >= pi && idx > 0 ? idx - 1 : 0))
  }

  // ---------- 保存 ----------
  const saveAll = async () => {
    const items: Record<string, unknown>[] = []
    const savedDrafts = new Set<DraftEdit>()
    photos.forEach(p => {
      const rel = p.result.image_url.replace(`${API_BASE}/uploads/`, '')
      p.drafts.forEach(d => {
        if (!d.checked || !d.name.trim()) return
        items.push({
          name: d.name.trim(),
          raw_name: d.raw_name,
          category_id: d.category_id,
          quantity: d.quantity,
          unit: d.unit,
          image_path: rel,
          bbox: JSON.stringify(d.bbox),
          production_date: d.production_date,
          shelf_life_days: d.shelf_life_days,
          expiry_date: d.expiry_date,
          date_text: d.date_text,
          date_source: d.date_source,
          confidence: d.confidence,
          needs_review: d.needs_review,
          note: d.note,
        })
        savedDrafts.add(d)
      })
    })
    if (items.length === 0) {
      toast.error('没有可保存的草稿（请填写物品名称）')
      return
    }
    setSaving(true)
    const tid = toast.loading(`正在保存 ${items.length} 条物品…`)
    try {
      const created = await api.createItems(items)
      setPhotos(prev => prev
        .map(p => ({ ...p, drafts: p.drafts.filter(d => !savedDrafts.has(d)) }))
        .filter(p => p.drafts.length > 0))
      if (activeIdx >= photos.length) setActiveIdx(0)
      toast.success(`已入库 ${created.length} 条物品`, { id: tid })
    } catch (e) {
      toast.error(`保存失败：${(e as Error).message}`, { id: tid })
    } finally {
      setSaving(false)
    }
  }

  const totalPending = photos.reduce((s, p) =>
    s + p.drafts.filter(d => d.checked && d.name.trim()).length, 0)

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-lg font-bold">拍照导入</h1>
          <p className="mt-1 text-sm text-slate-500">
            上传物品照片，系统自动完成 物品识别 → 自动归类 → 包装日期抽取 → 临期计算，确认后入库
          </p>
        </div>
        {totalPending > 0 && (
          <Button onClick={saveAll} disabled={saving} className="bg-sky-600 hover:bg-sky-700">
            {saving ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Save className="mr-2 h-4 w-4" />}
            保存 {totalPending} 条到清单
          </Button>
        )}
      </div>

      {/* 上传区 */}
      <div
        className={`rounded-xl border-2 border-dashed p-8 text-center transition-colors ${
          dragOver ? 'border-sky-400 bg-sky-50' : 'border-slate-300 bg-white hover:border-slate-400'
        }`}
        onDragOver={e => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={e => { e.preventDefault(); setDragOver(false); handleFiles(Array.from(e.dataTransfer.files)) }}
      >
        <input ref={inputRef} id="photo-input" type="file" accept="image/*" multiple
          style={{ position: 'absolute', width: 0, height: 0, opacity: 0, pointerEvents: 'none' }}
          onChange={e => { if (e.target.files) handleFiles(Array.from(e.target.files)); e.target.value = '' }} />
        <UploadCloud className="mx-auto h-10 w-10 text-slate-300" />
        <p className="mt-3 text-sm text-slate-600">
          拖拽照片到此处，或
          <button className="mx-1 font-medium text-sky-600 underline" onClick={() => inputRef.current?.click()}>
            点击选择（支持多张批量导入）
          </button>
        </p>
        <p className="mt-1 text-xs text-slate-400">支持 jpg / png / webp；识别在本地 CPU 完成，单张约 1-5 秒</p>
        {recognizing && (
          <div className="mt-4 flex items-center justify-center gap-2 text-sm text-sky-600">
            <Loader2 className="h-4 w-4 animate-spin" /> 模型推理中（YOLO11n 检测 + RapidOCR 文字识别 + CLIP 视觉分类）…
          </div>
        )}
      </div>

        {/* 照片队列 */}
        {photos.length > 0 && (
          <div className="flex gap-3 overflow-x-auto pb-2">
            {photos.map((p, i) => (
              <div key={i} className="relative shrink-0">
                <button onClick={() => setActiveIdx(i)}
                  className={`relative block h-20 w-20 overflow-hidden rounded-lg border-2 transition-all ${
                    i === activeIdx ? 'border-sky-500 ring-2 ring-sky-200' : 'border-slate-200 hover:border-slate-300'
                  }`}>
                  <img src={`${API_BASE}${p.result.image_url}`} alt="" className="h-full w-full object-cover" />
                  <span className="absolute bottom-1 right-1 rounded bg-black/60 px-1.5 text-[10px] text-white">
                    {p.drafts.length}
                  </span>
                  {p.drafts.length === 0 && (
                    <span className="absolute inset-0 flex items-center justify-center bg-emerald-500/80 text-xs font-medium text-white">已保存</span>
                  )}
                </button>
                <button onClick={() => removePhoto(i)} title="移除这张照片"
                  className="absolute -right-1.5 -top-1.5 flex h-5 w-5 items-center justify-center rounded-full bg-red-500 text-white shadow hover:bg-red-600">
                  <X className="h-3 w-3" />
                </button>
              </div>
            ))}
            <button onClick={() => inputRef.current?.click()}
              className="flex h-20 w-20 shrink-0 items-center justify-center rounded-lg border-2 border-dashed border-slate-300 text-slate-400 hover:border-sky-400 hover:text-sky-500">
              <ImagePlus className="h-6 w-6" />
            </button>
          </div>
        )}

      {/* 识别结果审核 */}
      {active && (
        <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
          {/* 左：原图 + 检测框 */}
          <Card className="overflow-hidden">
            <CardContent className="p-0">
              <div className="relative flex items-center justify-center bg-slate-900" style={{ minHeight: 320 }}>
                <div className="relative inline-block">
                  <img src={`${API_BASE}${active.result.image_url}`} alt="识别照片"
                    className="block max-h-[480px] w-auto" />
                  <svg className="pointer-events-none absolute inset-0 h-full w-full">
                    {active.drafts.map((d, di) => {
                      const [x1, y1, x2, y2] = d.bbox
                      if (!d.bbox.length) return null
                      return (
                        <g key={di}>
                          <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1}
                            fill={d.needs_review ? 'rgba(245,158,11,0.12)' : 'rgba(14,165,233,0.12)'}
                            stroke={d.needs_review ? '#f59e0b' : '#0ea5e9'} strokeWidth={2} rx={4} />
                          <text x={x1} y={Math.max(14, y1 - 6)} fontSize={13} fill="#fff"
                            stroke="rgba(0,0,0,0.65)" strokeWidth={3} paintOrder="stroke">
                            {d.name || '待确认'} {di + 1}
                          </text>
                        </g>
                      )
                    })}
                  </svg>
                </div>
              </div>
              <div className="flex items-center justify-between border-t border-slate-100 px-4 py-2 text-xs text-slate-400">
                <span>{active.result.filename} · {active.result.image_width}×{active.result.image_height}</span>
                <span>{active.drafts.length} 个识别区域</span>
              </div>
              {(active.result.ocr_texts?.length ?? 0) > 0 && (
                <div className="border-t border-slate-100 bg-slate-50 px-4 py-2">
                  <div className="mb-1 flex items-center gap-1 text-[11px] font-medium text-slate-400">
                    <Info className="h-3 w-3" /> OCR 识别到的文字
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {(active.result.ocr_texts ?? []).slice(0, 24).map((t, i) => (
                      <span key={i} className="rounded bg-white px-1.5 py-0.5 text-[11px] text-slate-500 ring-1 ring-slate-200">{t}</span>
                    ))}
                  </div>
                </div>
              )}
              {(active.result.clip_top?.length ?? 0) > 0 && (
                <div className="border-t border-slate-100 bg-violet-50/40 px-4 py-2">
                  <div className="mb-1 flex items-center gap-1 text-[11px] font-medium text-violet-400">
                    <Eye className="h-3 w-3" /> CLIP 视觉初判（零样本图文匹配，与 OCR 互相印证）
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {(active.result.clip_top ?? []).slice(0, 5).map((c, i) => (
                      <span key={i} className="rounded bg-white px-1.5 py-0.5 text-[11px] text-violet-600 ring-1 ring-violet-200">
                        {c.name} {Math.round(c.score * 100)}%
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* 右：草稿审核列表 */}
          <div className="space-y-3">
            {active.result.message && (
              <div className="flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-700">
                <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                {active.result.message}
              </div>
            )}
            {active.drafts.length === 0 && (
              <div className="rounded-lg border border-slate-200 bg-white py-10 text-center text-sm text-slate-400">
                这张照片的草稿已全部处理
              </div>
            )}
            {active.drafts.map((d, di) => (
              <Card key={di} className={d.needs_review ? 'border-amber-200' : ''}>
                <CardContent className="space-y-3 p-4">
                  {/* 头部：缩略裁剪 + 名称 + 置信度 */}
                  <div className="flex items-start gap-3">
                    {d.bbox.length === 4 && (
                      <div className="relative h-14 w-14 shrink-0 overflow-hidden rounded-lg bg-slate-100">
                        <BboxCrop imageUrl={`${API_BASE}${active.result.image_url}`}
                          bbox={d.bbox} imgW={active.result.image_width} imgH={active.result.image_height} />
                      </div>
                    )}
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <Input value={d.name}
                          placeholder="物品名称（可从下方候选点选）"
                          className="h-8 flex-1 text-sm font-medium"
                          onChange={e => patchDraft(activeIdx, di, { name: e.target.value })} />
                        {d.confidence !== null && (
                          <Badge variant="outline" className={d.confidence >= 0.6 ? 'border-emerald-200 bg-emerald-50 text-emerald-700' : 'border-amber-200 bg-amber-50 text-amber-700'}>
                            {Math.round(d.confidence * 100)}%
                          </Badge>
                        )}
                      </div>
                      <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[11px] text-slate-400">
                        {d.needs_review && <Badge variant="outline" className="border-amber-200 bg-amber-50 text-amber-600">待确认</Badge>}
                        {d.raw_name && <span>模型输出：{d.raw_name}</span>}
                        {d.date_text && <span className="max-w-[260px] truncate" title={d.date_text}>OCR：{d.date_text}</span>}
                      </div>
                    </div>
                    <Button variant="ghost" size="icon" className="h-8 w-8 text-slate-400 hover:text-red-600"
                      onClick={() => removeDraft(activeIdx, di)} title="丢弃此草稿">
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>

                  {/* 未识别出名称时的操作引导 */}
                  {!d.name.trim() && (
                    <div className="flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-700">
                      <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                      <span>
                        未能自动识别物品名称——请<b>点击下方候选物品</b>采用，或在上方输入框中手动输入。
                        {!d.production_date && !d.expiry_date && ' 包装上未找到日期文字，请在下方手动填写生产日期与保质期。'}
                      </span>
                    </div>
                  )}

                  {/* 候选点选 */}
                  {d.candidates.length > 0 && (
                    <div>
                      <div className="mb-1.5 flex items-center gap-1 text-[11px] font-medium text-slate-400">
                        <Crop className="h-3 w-3" /> 候选物品（点击采用）
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        {d.candidates.map((c, ci) => (
                          <button key={ci}
                            onClick={() => applyCandidate(activeIdx, di, c.name, c.category_id, c.category_name, c.default_shelf_life_days)}
                            className={`rounded-full border px-2.5 py-1 text-xs transition-colors ${
                              d.name === c.name
                                ? 'border-sky-400 bg-sky-50 text-sky-700'
                                : 'border-slate-200 bg-white text-slate-600 hover:border-sky-300 hover:text-sky-600'
                            }`}>
                            {c.name}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* 类别 / 数量 / 生产日期 */}
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                    <div>
                      <Label className="text-[11px] text-slate-400">类别</Label>
                      <Select value={String(d.category_id)}
                        onValueChange={v => {
                          const c = categories.find(x => x.id === Number(v))
                          patchDraft(activeIdx, di, { category_id: Number(v), category_name: c?.name ?? '' })
                        }}>
                        <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                        <SelectContent>
                          {categories.map(c => (
                            <SelectItem key={c.id} value={String(c.id)}>
                              <span className="flex items-center gap-1.5">
                                <span className="h-2 w-2 rounded-full" style={{ background: c.color }} />{c.name}
                              </span>
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label className="text-[11px] text-slate-400">数量</Label>
                      <div className="flex gap-1">
                        <Input type="number" min={1} value={d.quantity} className="h-8 w-14 text-xs"
                          onChange={e => patchDraft(activeIdx, di, { quantity: Math.max(1, Number(e.target.value)) })} />
                        <Input value={d.unit} className="h-8 flex-1 px-2 text-xs"
                          onChange={e => patchDraft(activeIdx, di, { unit: e.target.value })} />
                      </div>
                    </div>
                    <div>
                      <Label className="text-[11px] text-slate-400">生产日期</Label>
                      <Input type="date" value={d.production_date ?? ''} className="h-8 text-xs"
                        onChange={e => patchDraft(activeIdx, di, recomputeExpiry(d, { production_date: e.target.value || null }))} />
                    </div>
                  </div>

                  {/* 保质期 / 到期日 */}
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                    <div>
                      <Label className="text-[11px] text-slate-400">保质期（天）</Label>
                      <Input type="number" min={0} value={d.shelf_life_days ?? ''} className="h-8 text-xs"
                        onChange={e => patchDraft(activeIdx, di, recomputeExpiry(d, {
                          shelf_life_days: e.target.value ? Number(e.target.value) : null,
                        }))} />
                    </div>
                    <div>
                      <Label className="text-[11px] text-slate-400">到期日期</Label>
                      <Input type="date" value={d.expiry_date ?? ''} className="h-8 text-xs"
                        onChange={e => patchDraft(activeIdx, di, { expiry_date: e.target.value || null })} />
                    </div>
                    <div className="col-span-2 flex items-end">
                      <div className="text-[11px] leading-relaxed text-slate-400">
                        日期来源：<Badge variant="outline" className="ml-0.5 border-slate-200 text-[10px]">{DATE_SOURCE_LABEL[d.date_source] ?? d.date_source}</Badge>
                        {d.date_note && <div className="mt-0.5">{d.date_note}</div>}
                      </div>
                    </div>
                  </div>

                  {/* 操作 */}
                  <div className="flex items-center justify-between border-t border-slate-100 pt-2">
                    <label className="flex cursor-pointer items-center gap-1.5 text-xs text-slate-500">
                      <input type="checkbox" checked={d.checked} className="accent-sky-600"
                        onChange={e => patchDraft(activeIdx, di, { checked: e.target.checked })} />
                      并入本次保存
                    </label>
                    <Button size="sm" variant="outline" className="h-7 text-xs"
                      disabled={!d.name.trim()}
                      onClick={async () => {
                        try {
                          const rel = active.result.image_url.replace(`${API_BASE}/uploads/`, '')
                          const created = await api.createItems([{
                            name: d.name.trim(), raw_name: d.raw_name, category_id: d.category_id,
                            quantity: d.quantity, unit: d.unit,
                            image_path: rel, bbox: JSON.stringify(d.bbox),
                            production_date: d.production_date, shelf_life_days: d.shelf_life_days,
                            expiry_date: d.expiry_date, date_text: d.date_text,
                            date_source: d.date_source, confidence: d.confidence,
                            needs_review: d.needs_review, note: d.note,
                          }])
                          removeDraft(activeIdx, di)
                          toast.success(`「${created[0].name}」已入库`)
                        } catch (e) {
                          toast.error(`入库失败：${(e as Error).message}`)
                        }
                      }}>
                      <Check className="mr-1 h-3.5 w-3.5" /> 单条入库
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}

      {photos.length === 0 && (
        <div className="rounded-xl border border-slate-200 bg-white px-6 py-10 text-center">
          <h3 className="text-sm font-medium text-slate-600">使用流程</h3>
          <div className="mx-auto mt-4 grid max-w-3xl grid-cols-2 gap-3 text-xs text-slate-500 sm:grid-cols-4">
            {[
              ['拍照 / 选图', '物品照片，可多选批量'],
              ['识别与归类', 'YOLO 检测 + OCR 读包装文字 + CLIP 视觉初判，三路互补'],
              ['审核草稿', '低置信度给候选点选，日期可手动修正'],
              ['入库与看板', '保存后进入临期看板，可导出表格'],
            ].map(([t, d], i) => (
              <div key={i} className="rounded-lg bg-slate-50 p-3">
                <div className="mb-1 flex h-6 w-6 items-center justify-center rounded-full bg-sky-100 text-[11px] font-bold text-sky-600">{i + 1}</div>
                <div className="font-medium text-slate-600">{t}</div>
                <div className="mt-0.5 text-[11px] leading-relaxed">{d}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {photos.length > 0 && photos.every(p => p.drafts.length === 0) && (
        <div className="text-center">
          <Button variant="outline" onClick={() => navigate('/items')}>前往物品清单查看 →</Button>
        </div>
      )}
    </div>
  )
}

/** 按 bbox 从原图裁剪的缩略展示 */
function BboxCrop({ imageUrl, bbox, imgW, imgH }: {
  imageUrl: string; bbox: number[]; imgW: number; imgH: number
}) {
  const [x1, y1, x2, y2] = bbox
  const size = 56
  const bw = Math.max(1, x2 - x1)
  const bh = Math.max(1, y2 - y1)
  const scale = size / Math.max(bw, bh)
  return (
    <img src={imageUrl} alt=""
      style={{
        position: 'absolute',
        width: imgW * scale,
        height: imgH * scale,
        left: -x1 * scale,
        top: -y1 * scale,
        maxWidth: 'none',
      }} />
  )
}
