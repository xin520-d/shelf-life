/** API 客户端与类型定义（与后端 schemas 对应） */

// ---------- 类型 ----------
export interface Category {
  id: number
  name: string
  color: string
  default_shelf_life_days: number
  is_builtin: boolean
  item_count: number
}

export interface Candidate {
  name: string
  category_id: number
  category_name: string
  default_shelf_life_days: number
}

export interface DetectionDraft {
  name: string
  raw_name: string
  confidence: number | null
  bbox: number[]
  category_id: number
  category_name: string
  quantity: number
  unit: string
  production_date: string | null
  shelf_life_days: number | null
  expiry_date: string | null
  date_text: string
  date_source: string
  date_note: string
  needs_review: boolean
  candidates: Candidate[]
  note: string
}

export interface PhotoResult {
  filename: string
  image_url: string
  image_width: number
  image_height: number
  ocr_texts: string[]
  drafts: DetectionDraft[]
  message: string
  clip_top: { name: string; category: string; score: number }[]
  is_screenshot: boolean
}

export interface ModelInfo {
  name: string
  ready: boolean
  desc: string
  download_state?: string
  download_error?: string
}

export type ExpiryStatus = 'normal' | 'near' | 'expired' | 'unknown'

export interface Item {
  id: number
  name: string
  raw_name: string
  category_id: number
  category_name: string
  category_color: string
  quantity: number
  unit: string
  image_path: string
  thumb_url: string | null
  production_date: string | null
  shelf_life_days: number | null
  expiry_date: string | null
  date_text: string
  date_source: string
  confidence: number | null
  needs_review: boolean
  note: string
  status: string
  remaining_days: number | null
  expiry_status: ExpiryStatus
  created_at: string | null
}

export interface Stats {
  total_active: number
  near_expiry: number
  expired: number
  normal: number
  unknown: number
  due_7d: number
  due_30d: number
  by_category: { name: string; color: string; count: number }[]
  expiry_timeline: { date: string; count: number }[]
  recent_items: Item[]
}

export interface Correction {
  id: number
  raw_name: string
  item_name: string
  category_name: string
  use_count: number
  last_used_at: string | null
}

// ---------- 请求封装 ----------
/** 把 FastAPI 的 422 校验错误数组转成人能看懂的一句话 */
function formatDetail(detail: unknown): string {
  if (Array.isArray(detail)) {
    const parts = detail.map((d) => {
      const err = d as { loc?: unknown[]; msg?: string }
      const field = Array.isArray(err.loc) ? err.loc.filter(x => x !== 'query' && x !== 'body').join('.') : ''
      return field ? `${field}：${err.msg ?? '参数不合法'}` : (err.msg ?? '参数不合法')
    })
    return parts.join('；') || '请求参数不合法'
  }
  try {
    return JSON.stringify(detail)
  } catch {
    return String(detail)
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {}
  // 仅对 JSON 字符串体设置 Content-Type；FormData 由浏览器自动携带 boundary
  if (options?.body && typeof options.body === 'string') {
    headers['Content-Type'] = 'application/json'
  }
  const res = await fetch(path, { ...options, headers })
  if (!res.ok) {
    let msg = `请求失败 (${res.status})`
    try {
      const data = await res.json()
      if (data.detail) msg = typeof data.detail === 'string' ? data.detail : formatDetail(data.detail)
    } catch { /* ignore */ }
    throw new Error(msg)
  }
  return res.json() as Promise<T>
}

export const api = {
  // 识别
  recognize: (files: File[]) => {
    const fd = new FormData()
    files.forEach(f => fd.append('files', f))
    return request<PhotoResult[]>('/api/recognize', { method: 'POST', body: fd })
  },
  // 物品
  createItems: (items: unknown[]) =>
    request<Item[]>('/api/items', { method: 'POST', body: JSON.stringify({ items }) }),
  listItems: (params: Record<string, string | number | undefined> = {}) => {
    const q = new URLSearchParams()
    Object.entries(params).forEach(([k, v]) => { if (v !== undefined && v !== '') q.set(k, String(v)) })
    const s = q.toString()
    return request<Item[]>(`/api/items${s ? '?' + s : ''}`)
  },
  // 分页版列表：返回本页数据与总条数（从响应头 X-Total-Count 读取）
  listItemsPaged: async (params: Record<string, string | number | undefined> = {}) => {
    const q = new URLSearchParams()
    Object.entries(params).forEach(([k, v]) => { if (v !== undefined && v !== '') q.set(k, String(v)) })
    const s = q.toString()
    const res = await fetch(`/api/items${s ? '?' + s : ''}`)
    if (!res.ok) {
      let msg = `请求失败 (${res.status})`
      try {
        const data = await res.json()
        if (data.detail) msg = typeof data.detail === 'string' ? data.detail : formatDetail(data.detail)
      } catch { /* ignore */ }
      throw new Error(msg)
    }
    const items = await res.json() as Item[]
    const total = Number(res.headers.get('X-Total-Count') ?? items.length)
    return { items, total }
  },
  batchConfirm: (ids: number[]) =>
    request<{ updated: number }>('/api/items/batch/confirm', { method: 'POST', body: JSON.stringify({ ids }) }),
  batchFinish: (ids: number[], action: 'consume' | 'discard') =>
    request<{ updated: number; status: string }>('/api/items/batch/finish', { method: 'POST', body: JSON.stringify({ ids, action }) }),
  batchDelete: (ids: number[]) =>
    request<{ deleted: number }>('/api/items/batch/delete', { method: 'POST', body: JSON.stringify({ ids }) }),
  patchItem: (id: number, data: Record<string, unknown>) =>
    request<Item>(`/api/items/${id}`, { method: 'PATCH', body: JSON.stringify(data) }),
  finishItem: (id: number, action: 'consume' | 'discard') =>
    request<{ ok: boolean; status: string }>(`/api/items/${id}/finish?action=${action}`, { method: 'POST' }),
  restoreItem: (id: number) =>
    request<{ ok: boolean }>(`/api/items/${id}/restore`, { method: 'POST' }),
  deleteItem: (id: number) =>
    request<{ ok: boolean }>(`/api/items/${id}`, { method: 'DELETE' }),
  // 元数据
  categories: () => request<Category[]>('/api/categories'),
  createCategory: (data: { name: string; color: string; default_shelf_life_days: number }) =>
    request<Category>('/api/categories', { method: 'POST', body: JSON.stringify(data) }),
  updateCategory: (id: number, data: { name: string; color: string; default_shelf_life_days: number }) =>
    request<Category>(`/api/categories/${id}`, { method: 'PATCH', body: JSON.stringify(data) }),
  deleteCategory: (id: number) =>
    request<{ ok: boolean }>(`/api/categories/${id}`, { method: 'DELETE' }),
  settings: () => request<{ near_expiry_days: number; low_conf: number }>('/api/settings'),
  updateSettings: (payload: { near_expiry_days: number; low_conf: number }) =>
    request<{ near_expiry_days: number; low_conf: number }>('/api/settings', { method: 'PUT', body: JSON.stringify(payload) }),
  corrections: () => request<Correction[]>('/api/corrections'),
  deleteCorrection: (id: number) =>
    request<{ ok: boolean }>(`/api/corrections/${id}`, { method: 'DELETE' }),
  // 识别模型状态与安装
  models: () => request<Record<'yolo' | 'ocr' | 'clip', ModelInfo>>('/api/models'),
  downloadClip: () =>
    request<{ ok: boolean; message: string }>('/api/models/clip/download', { method: 'POST' }),
  // 统计与导出
  stats: () => request<Stats>('/api/stats'),
  exportCsv: (params: Record<string, string> = {}) => {
    const q = new URLSearchParams(params).toString()
    return `/api/export/csv${q ? '?' + q : ''}`
  },
  exportXlsx: (params: Record<string, string> = {}) => {
    const q = new URLSearchParams(params).toString()
    return `/api/export/xlsx${q ? '?' + q : ''}`
  },
}

// ---------- 展示辅助 ----------
export const STATUS_LABEL: Record<ExpiryStatus, string> = {
  normal: '正常',
  near: '临期',
  expired: '已过期',
  unknown: '未知',
}

export const STATUS_STYLE: Record<ExpiryStatus, string> = {
  normal: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  near: 'bg-amber-50 text-amber-700 border-amber-200',
  expired: 'bg-red-50 text-red-700 border-red-200',
  unknown: 'bg-slate-100 text-slate-500 border-slate-200',
}

export const DATE_SOURCE_LABEL: Record<string, string> = {
  ocr: 'OCR 识别',
  manual: '人工填写',
  category_default: '类别默认',
}

export function remainingText(days: number | null): string {
  if (days === null) return '—'
  if (days < 0) return `已过期 ${-days} 天`
  if (days === 0) return '今天到期'
  return `剩 ${days} 天`
}
