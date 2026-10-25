import { useCallback, useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { toast } from 'sonner'
import {
  Search, Pencil, Trash2, Utensils, Ban, RotateCcw, PackageOpen, SlidersHorizontal,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Card } from '@/components/ui/card'
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from '@/components/ui/select'
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter,
} from '@/components/ui/dialog'
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from '@/components/ui/table'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { api, type Item, STATUS_STYLE, STATUS_LABEL, DATE_SOURCE_LABEL, remainingText } from '@/lib/api'
import { useMeta } from '@/hooks/useMeta'

const API_BASE = ''

const TABS = [
  { value: 'active', label: '在册' },
  { value: 'near', label: '临期' },
  { value: 'expired', label: '已过期' },
  { value: 'finished', label: '已消耗/丢弃' },
  { value: 'all', label: '全部' },
]

export default function ItemsPage() {
  const { categories } = useMeta()
  const [searchParams] = useSearchParams()
  const [items, setItems] = useState<Item[]>([])
  // 支持 /items?tab=near 直接落到临期（看板卡片跳转用）
  const [tab, setTab] = useState(() => {
    const t = searchParams.get('tab')
    return TABS.some(x => x.value === t) ? (t as string) : 'active'
  })
  const [q, setQ] = useState('')
  const [catFilter, setCatFilter] = useState('')
  const [loading, setLoading] = useState(true)
  const [editing, setEditing] = useState<Item | null>(null)
  const [deleting, setDeleting] = useState<Item | null>(null)
  // 分页与批量选择
  const [page, setPage] = useState(1)
  const [size, setSize] = useState(20)
  const [total, setTotal] = useState(0)
  const [selected, setSelected] = useState<Set<number>>(new Set())

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const params: Record<string, string | number | undefined> = { sort: 'expiry', page, size }
      if (tab === 'active') params.status = 'active'
      // 临期 / 已过期都按「有效期状态」过滤（后端用设置里的临期天数判定），
      // 不能用「N 天内」——那会把剩余天数为负的已过期物品全算进来。
      else if (tab === 'near') { params.status = 'active'; params.expiry_status = 'near' }
      else if (tab === 'expired') { params.status = 'active'; params.expiry_status = 'expired' }
      else if (tab === 'finished') params.status = 'consumed'
      else params.status = 'all'
      // 注意：下拉里"全部"的 value 是字符串 'all'，不能当 ID 传给后端（否则 422）
      if (catFilter && catFilter !== 'all') params.category_id = catFilter
      if (q) params.q = q
      const [{ items: data, total: t }] = await Promise.all([
        api.listItemsPaged(params),
        api.stats().then(s => setCounts({
          active: s.total_active, near: s.near_expiry, expired: s.expired,
        })).catch(() => undefined),
      ])
      setItems(data)
      setTotal(t)
    } catch (e) {
      toast.error((e as Error).message)
    } finally {
      setLoading(false)
    }
  }, [tab, q, catFilter, page, size])

  // 筛选条件变化时回到第 1 页
  useEffect(() => { setPage(1) }, [tab, q, catFilter])

  const [counts, setCounts] = useState({ active: 0, near: 0, expired: 0 })

  const toggleSelect = (id: number) => {
    setSelected(prev => {
      const n = new Set(prev)
      if (n.has(id)) n.delete(id); else n.add(id)
      return n
    })
  }
  const toggleSelectAllOnPage = () => {
    const pageIds = items.map(i => i.id)
    const allOnPage = pageIds.length > 0 && pageIds.every(id => selected.has(id))
    setSelected(prev => {
      const n = new Set(prev)
      pageIds.forEach(id => { if (allOnPage) n.delete(id); else n.add(id) })
      return n
    })
  }
  const clearSelection = () => setSelected(new Set())

  useEffect(() => { load() }, [load])

  const finish = async (item: Item, action: 'consume' | 'discard') => {
    await api.finishItem(item.id, action)
    toast.success(`「${item.name}」已登记为${action === 'consume' ? '消耗' : '丢弃'}`)
    load()
  }

  const restore = async (item: Item) => {
    await api.restoreItem(item.id)
    toast.success(`「${item.name}」已恢复到在册`)
    load()
  }

  const remove = async (item: Item) => {
    try {
      await api.deleteItem(item.id)
      toast.success(`「${item.name}」已删除`)
      setDeleting(null)
      load()
    } catch (e) {
      toast.error((e as Error).message)
    }
  }

  const [batchDeleting, setBatchDeleting] = useState(false)

  const runBatchConfirm = async () => {
    try {
      const r = await api.batchConfirm(Array.from(selected))
      toast.success(`已确认 ${r.updated} 件`)
      clearSelection(); load()
    } catch (e) { toast.error((e as Error).message) }
  }
  const runBatchFinish = async (action: 'consume' | 'discard') => {
    try {
      const r = await api.batchFinish(Array.from(selected), action)
      toast.success(`已批量${action === 'consume' ? '登记消耗' : '登记丢弃'} ${r.updated} 件（可恢复）`)
      clearSelection(); load()
    } catch (e) { toast.error((e as Error).message) }
  }
  const runBatchDelete = async () => {
    try {
      const r = await api.batchDelete(Array.from(selected))
      toast.success(`已删除 ${r.deleted} 件`)
      setBatchDeleting(false); clearSelection(); load()
    } catch (e) { toast.error((e as Error).message) }
  }

  const tabCount = counts

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-lg font-bold">物品清单</h1>
          <p className="mt-1 text-sm text-slate-500">管理在册物品：编辑、消耗登记、丢弃与恢复</p>
        </div>
      </div>

      {/* 过滤栏 */}
      <div className="flex flex-wrap items-center gap-2">
        <Tabs value={tab} onValueChange={setTab}>
          <TabsList>
            {TABS.map(t => (
              <TabsTrigger key={t.value} value={t.value} className="text-xs">
                {t.label}
                {t.value === 'near' && tabCount.near > 0 && (
                  <span className="ml-1 rounded-full bg-amber-100 px-1.5 text-[10px] text-amber-700">{tabCount.near}</span>
                )}
                {t.value === 'expired' && tabCount.expired > 0 && (
                  <span className="ml-1 rounded-full bg-red-100 px-1.5 text-[10px] text-red-700">{tabCount.expired}</span>
                )}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
        <div className="flex flex-1 flex-wrap items-center justify-end gap-2">
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-400" />
            <Input placeholder="搜索名称 / 备注" value={q} onChange={e => setQ(e.target.value)}
              className="h-9 w-44 pl-8 text-sm" />
          </div>
          <Select value={catFilter} onValueChange={setCatFilter}>
            <SelectTrigger className="h-9 w-28 text-xs"><SelectValue placeholder="全部类别" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">全部类别</SelectItem>
              {categories.map(c => <SelectItem key={c.id} value={String(c.id)}>{c.name}</SelectItem>)}
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* 批量操作栏：选中后显示 */}
      {selected.size > 0 && (
        <div className="flex flex-wrap items-center gap-2 rounded-lg border border-sky-200 bg-sky-50 px-3 py-2">
          <span className="text-sm font-medium text-sky-800">已选 {selected.size} 件</span>
          <div className="ml-auto flex flex-wrap gap-1.5">
            <Button size="sm" variant="outline" className="border-sky-300 text-sky-700" onClick={runBatchConfirm}>批量确认</Button>
            <Button size="sm" variant="outline" className="border-emerald-300 text-emerald-700" onClick={() => runBatchFinish('consume')}>批量登记消耗</Button>
            <Button size="sm" variant="outline" className="border-amber-300 text-amber-700" onClick={() => runBatchFinish('discard')}>批量登记丢弃</Button>
            <Button size="sm" variant="destructive" onClick={() => setBatchDeleting(true)}>批量删除</Button>
            <Button size="sm" variant="ghost" onClick={clearSelection}>取消选择</Button>
          </div>
        </div>
      )}

      {/* 列表 */}
      <Card>
        {loading ? (
          <div className="py-16 text-center text-sm text-slate-400">加载中…</div>
        ) : items.length === 0 ? (
          <div className="flex flex-col items-center py-16 text-slate-400">
            <PackageOpen className="h-10 w-10" />
            <p className="mt-3 text-sm">
              {tab === 'near' ? '没有临期物品（已过期的请切到「已过期」标签）' : '没有符合条件的物品'}
            </p>
          </div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow className="bg-slate-50/60">
                <TableHead className="w-10">
                  <Checkbox checked={items.length > 0 && items.every(i => selected.has(i.id))}
                    onCheckedChange={toggleSelectAllOnPage} aria-label="全选本页" />
                </TableHead>
                <TableHead className="w-14">图片</TableHead>
                <TableHead>物品</TableHead>
                <TableHead>类别</TableHead>
                <TableHead className="text-center">数量</TableHead>
                <TableHead>生产日期</TableHead>
                <TableHead>到期日期</TableHead>
                <TableHead>剩余</TableHead>
                <TableHead>状态</TableHead>
                <TableHead className="text-right">操作</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {items.map(it => (
                <TableRow key={it.id} className={it.status !== 'active' ? 'opacity-55' : ''}>
                  <TableCell>
                    <Checkbox checked={selected.has(it.id)} onCheckedChange={() => toggleSelect(it.id)} aria-label="选择" />
                  </TableCell>
                  <TableCell>
                    {it.thumb_url ? (
                      <img src={`${API_BASE}${it.thumb_url}`} alt=""
                        className="h-10 w-10 rounded-md object-cover ring-1 ring-slate-200" />
                    ) : (
                      <div className="flex h-10 w-10 items-center justify-center rounded-md bg-slate-100 text-slate-300">
                        <PackageOpen className="h-4 w-4" />
                      </div>
                    )}
                  </TableCell>
                  <TableCell>
                    <div className="font-medium">{it.name}</div>
                    <div className="text-[11px] text-slate-400">
                      {it.raw_name && <span className="mr-1.5">{it.raw_name}</span>}
                      {it.confidence !== null && <span className="mr-1.5">{Math.round(it.confidence * 100)}%</span>}
                      {DATE_SOURCE_LABEL[it.date_source] ?? it.date_source}
                      {it.needs_review && <Badge variant="outline" className="ml-1 border-amber-200 text-[10px] text-amber-600">待确认</Badge>}
                    </div>
                  </TableCell>
                  <TableCell>
                    <span className="flex items-center gap-1.5 text-sm">
                      <span className="h-2 w-2 rounded-full" style={{ background: it.category_color }} />
                      {it.category_name}
                    </span>
                  </TableCell>
                  <TableCell className="text-center text-sm tabular-nums">{it.quantity}{it.unit}</TableCell>
                  <TableCell className="text-sm tabular-nums text-slate-500">{it.production_date ?? '—'}</TableCell>
                  <TableCell className="text-sm tabular-nums text-slate-500">{it.expiry_date ?? '—'}</TableCell>
                  <TableCell className="text-sm tabular-nums">{remainingText(it.remaining_days)}</TableCell>
                  <TableCell>
                    {it.status !== 'active' ? (
                      <Badge variant="outline" className="border-slate-200 bg-slate-50 text-slate-500">
                        {it.status === 'consumed' ? '已消耗' : '已丢弃'}
                      </Badge>
                    ) : (
                      <Badge variant="outline" className={STATUS_STYLE[it.expiry_status]}>
                        {STATUS_LABEL[it.expiry_status]}
                      </Badge>
                    )}
                  </TableCell>
                  <TableCell className="text-right">
                    <div className="flex justify-end gap-0.5">
                      {it.status === 'active' ? (
                        <>
                          <Button variant="ghost" size="icon" className="h-7 w-7" title="编辑"
                            onClick={() => setEditing(it)}><Pencil className="h-3.5 w-3.5" /></Button>
                          <Button variant="ghost" size="icon" className="h-7 w-7 text-emerald-600" title="登记消耗"
                            onClick={() => finish(it, 'consume')}><Utensils className="h-3.5 w-3.5" /></Button>
                          <Button variant="ghost" size="icon" className="h-7 w-7 text-amber-600" title="登记丢弃"
                            onClick={() => finish(it, 'discard')}><Ban className="h-3.5 w-3.5" /></Button>
                        </>
                      ) : (
                        <Button variant="ghost" size="icon" className="h-7 w-7" title="恢复到在册"
                          onClick={() => restore(it)}><RotateCcw className="h-3.5 w-3.5" /></Button>
                      )}
                      <Button variant="ghost" size="icon" className="h-7 w-7 text-red-500" title="删除"
                        onClick={() => setDeleting(it)}><Trash2 className="h-3.5 w-3.5" /></Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </Card>

      {/* 分页 */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <span>每页</span>
          <Select value={String(size)} onValueChange={v => { setSize(Number(v)); setPage(1) }}>
            <SelectTrigger className="h-8 w-16 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>
              {[10, 20, 50, 100].map(n => <SelectItem key={n} value={String(n)}>{n}</SelectItem>)}
            </SelectContent>
          </Select>
          <span>条</span>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <span>共 {total} 条 · 第 {Math.min(page, Math.max(1, Math.ceil(total / size)))} / {Math.max(1, Math.ceil(total / size))} 页</span>
          <Button variant="outline" size="sm" className="h-8" disabled={page <= 1} onClick={() => setPage(p => p - 1)}>上一页</Button>
          <Button variant="outline" size="sm" className="h-8" disabled={page >= Math.max(1, Math.ceil(total / size))} onClick={() => setPage(p => p + 1)}>下一页</Button>
        </div>
      </div>

      <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
        <SlidersHorizontal className="h-3 w-3" />
        共 {total} 条 · 修改类别/名称会自动记忆，下次识别同源物品时优先沿用
      </div>

      {deleting && (
        <div
          className="fixed inset-0 z-[200] flex items-center justify-center bg-black/50 p-4"
          onClick={() => setDeleting(null)}
        >
          <div
            className="w-full max-w-sm rounded-lg border bg-white p-6 shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="text-lg font-semibold text-slate-900">删除物品</h2>
            <p className="mt-2 text-sm text-slate-600">
              确定彻底删除「{deleting.name}」吗？此操作不可恢复。
            </p>
            <div className="mt-5 flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
              <Button variant="outline" onClick={() => setDeleting(null)}>取消</Button>
              <Button variant="destructive" onClick={() => remove(deleting)}>删除</Button>
            </div>
          </div>
        </div>
      )}

      {editing && (
        <EditDialog item={editing} categories={categories}
          onClose={() => setEditing(null)} onSaved={() => { setEditing(null); load() }} />
      )}

      {batchDeleting && (
        <div
          className="fixed inset-0 z-[200] flex items-center justify-center bg-black/50 p-4"
          onClick={() => setBatchDeleting(false)}
        >
          <div
            className="w-full max-w-sm rounded-lg border bg-white p-6 shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="text-lg font-semibold text-slate-900">批量删除</h2>
            <p className="mt-2 text-sm text-slate-600">
              确定彻底删除选中的 <span className="font-semibold text-red-600">{selected.size}</span> 件物品吗？此操作不可恢复。
            </p>
            <div className="mt-5 flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
              <Button variant="outline" onClick={() => setBatchDeleting(false)}>取消</Button>
              <Button variant="destructive" onClick={runBatchDelete}>删除</Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function EditDialog({ item, categories, onClose, onSaved }: {
  item: Item
  categories: { id: number; name: string; color: string }[]
  onClose: () => void
  onSaved: () => void
}) {
  const [form, setForm] = useState({
    name: item.name,
    category_id: item.category_id,
    quantity: item.quantity,
    unit: item.unit,
    production_date: item.production_date ?? '',
    shelf_life_days: item.shelf_life_days ?? '',
    expiry_date: item.expiry_date ?? '',
    note: item.note,
  })
  const [saving, setSaving] = useState(false)

  const recompute = (patch: Partial<typeof form>) => {
    const next = { ...form, ...patch }
    if (next.production_date && next.shelf_life_days && Number(next.shelf_life_days) > 0) {
      const d = new Date(next.production_date)
      d.setDate(d.getDate() + Number(next.shelf_life_days))
      next.expiry_date = d.toISOString().slice(0, 10)
    }
    setForm(next)
  }

  const save = async () => {
    setSaving(true)
    try {
      await api.patchItem(item.id, {
        name: form.name,
        category_id: form.category_id,
        quantity: Number(form.quantity),
        unit: form.unit,
        production_date: form.production_date || null,
        shelf_life_days: form.shelf_life_days ? Number(form.shelf_life_days) : null,
        expiry_date: form.expiry_date || null,
        date_source: 'manual',
        note: form.note,
        needs_review: false,
      })
      toast.success(`「${form.name}」已更新`)
      onSaved()
    } catch (e) {
      toast.error((e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog open onOpenChange={onClose}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>编辑物品</DialogTitle>
        </DialogHeader>
        <div className="grid grid-cols-2 gap-3">
          <div className="col-span-2">
            <Label className="text-xs">名称</Label>
            <Input value={form.name} onChange={e => setForm({ ...form, name: e.target.value })} />
          </div>
          <div>
            <Label className="text-xs">类别</Label>
            <Select value={String(form.category_id)}
              onValueChange={v => setForm({ ...form, category_id: Number(v) })}>
              <SelectTrigger><SelectValue /></SelectTrigger>
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
            <Label className="text-xs">数量</Label>
            <div className="flex gap-1">
              <Input type="number" min={1} value={form.quantity}
                onChange={e => setForm({ ...form, quantity: Number(e.target.value) })} className="w-20" />
              <Input value={form.unit} onChange={e => setForm({ ...form, unit: e.target.value })} />
            </div>
          </div>
          <div>
            <Label className="text-xs">生产日期</Label>
            <Input type="date" value={form.production_date}
              onChange={e => recompute({ production_date: e.target.value })} />
          </div>
          <div>
            <Label className="text-xs">保质期（天）</Label>
            <Input type="number" min={0} value={form.shelf_life_days}
              onChange={e => recompute({ shelf_life_days: e.target.value })} />
          </div>
          <div>
            <Label className="text-xs">到期日期</Label>
            <Input type="date" value={form.expiry_date}
              onChange={e => setForm({ ...form, expiry_date: e.target.value })} />
          </div>
          <div className="col-span-2">
            <Label className="text-xs">备注</Label>
            <Input value={form.note} onChange={e => setForm({ ...form, note: e.target.value })} />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>取消</Button>
          <Button onClick={save} disabled={saving || !form.name.trim()} className="bg-sky-600 hover:bg-sky-700">
            {saving ? '保存中…' : '保存'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
