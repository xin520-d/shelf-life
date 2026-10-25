import { useCallback, useEffect, useState } from 'react'
import { toast } from 'sonner'
import { Download, FileSpreadsheet, FileText, Search } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent } from '@/components/ui/card'
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from '@/components/ui/select'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { api, type Item, STATUS_STYLE, STATUS_LABEL, remainingText } from '@/lib/api'
import { useMeta } from '@/hooks/useMeta'

export default function ExportPage() {
  const { categories } = useMeta()
  const [items, setItems] = useState<Item[]>([])
  const [status, setStatus] = useState('all')
  const [catFilter, setCatFilter] = useState('')
  const [q, setQ] = useState('')
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const params: Record<string, string> = { sort: 'expiry', status, size: '200' }
      // "全部"的 value 是字符串 'all'，不能当 ID 传
      if (catFilter && catFilter !== 'all') params.category_id = catFilter
      if (q) params.q = q
      setItems(await api.listItems(params))
    } finally {
      setLoading(false)
    }
  }, [status, catFilter, q])

  useEffect(() => { load() }, [load])

  const download = (url: string, name: string) => {
    const a = document.createElement('a')
    a.href = url
    a.download = name
    document.body.appendChild(a)
    a.click()
    a.remove()
    toast.success(`已开始下载 ${name}`)
  }

  const now = new Date()
  const stamp = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}`

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-lg font-bold">数据导出</h1>
        <p className="mt-1 text-sm text-slate-500">
          把当前筛选范围内的保质期清单导出为表格文件，便于线下核对与分享（XLSX 含汇总 sheet，临期/过期自动着色）
        </p>
      </div>

      {/* 筛选 */}
      <Card>
        <CardContent className="flex flex-wrap items-end gap-3 p-4">
          <div>
            <div className="mb-1 text-xs text-slate-400">范围</div>
            <Select value={status} onValueChange={setStatus}>
              <SelectTrigger className="h-9 w-32 text-sm"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="active">在册物品</SelectItem>
                <SelectItem value="consumed">已消耗</SelectItem>
                <SelectItem value="discarded">已丢弃</SelectItem>
                <SelectItem value="all">全部记录</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div>
            <div className="mb-1 text-xs text-slate-400">类别</div>
            <Select value={catFilter} onValueChange={setCatFilter}>
              <SelectTrigger className="h-9 w-28 text-sm"><SelectValue placeholder="全部" /></SelectTrigger>
              <SelectContent>
                <SelectItem value="all">全部</SelectItem>
                {categories.map(c => <SelectItem key={c.id} value={String(c.id)}>{c.name}</SelectItem>)}
              </SelectContent>
            </Select>
          </div>
          <div>
            <div className="mb-1 text-xs text-slate-400">搜索</div>
            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-400" />
              <Input placeholder="名称关键词" value={q} onChange={e => setQ(e.target.value)}
                className="h-9 w-40 pl-8 text-sm" />
            </div>
          </div>
          <div className="ml-auto flex gap-2">
            <Button variant="outline" onClick={() => {
              const params: Record<string, string> = { status }
              if (catFilter && catFilter !== 'all') params.category_id = catFilter
              if (q) params.q = q
              download(api.exportCsv(params), `保质期清单_${stamp}.csv`)
            }}>
              <FileText className="mr-2 h-4 w-4" />导出 CSV
            </Button>
            <Button className="bg-emerald-600 hover:bg-emerald-700" onClick={() => {
              const params: Record<string, string> = { status }
              if (catFilter && catFilter !== 'all') params.category_id = catFilter
              if (q) params.q = q
              download(api.exportXlsx(params), `保质期清单_${stamp}.xlsx`)
            }}>
              <FileSpreadsheet className="mr-2 h-4 w-4" />导出 XLSX
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* 预览 */}
      <Card>
        <CardContent className="p-0">
          <div className="flex items-center justify-between border-b border-slate-100 px-4 py-2.5">
            <span className="text-sm font-medium">导出预览</span>
            <span className="text-xs text-slate-400">共 {items.length} 条</span>
          </div>
          {loading ? (
            <div className="py-16 text-center text-sm text-slate-400">加载中…</div>
          ) : items.length === 0 ? (
            <div className="py-16 text-center text-sm text-slate-400">当前范围没有数据</div>
          ) : (
            <div className="max-h-[480px] overflow-auto">
              <Table>
                <TableHeader className="sticky top-0 bg-white">
                  <TableRow>
                    <TableHead>物品名称</TableHead><TableHead>类别</TableHead>
                    <TableHead>数量</TableHead><TableHead>生产日期</TableHead><TableHead>保质期(天)</TableHead>
                    <TableHead>到期日期</TableHead><TableHead>剩余天数</TableHead><TableHead>状态</TableHead>
                    <TableHead>日期来源</TableHead><TableHead>备注</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {items.map(it => (
                    <TableRow key={it.id}>
                      <TableCell className="font-medium">{it.name}</TableCell>
                      <TableCell>{it.category_name}</TableCell>
                      <TableCell>{it.quantity}{it.unit}</TableCell>
                      <TableCell className="tabular-nums">{it.production_date ?? '—'}</TableCell>
                      <TableCell className="tabular-nums">{it.shelf_life_days ?? '—'}</TableCell>
                      <TableCell className="tabular-nums">{it.expiry_date ?? '—'}</TableCell>
                      <TableCell className="tabular-nums">{remainingText(it.remaining_days)}</TableCell>
                      <TableCell>
                        <span className={`rounded border px-1.5 py-0.5 text-[11px] ${STATUS_STYLE[it.expiry_status]}`}>
                          {STATUS_LABEL[it.expiry_status]}
                        </span>
                      </TableCell>
                      <TableCell className="text-xs text-slate-500">
                        {it.date_source === 'ocr' ? 'OCR识别' : it.date_source === 'manual' ? '人工填写' : '类别默认'}
                      </TableCell>
                      <TableCell className="max-w-[160px] truncate text-xs text-slate-500" title={it.note}>{it.note || '—'}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="flex items-center gap-2 rounded-lg border border-slate-200 bg-white p-4 text-xs text-slate-500">
        <Download className="h-4 w-4 text-slate-400" />
        CSV 为 UTF-8-SIG 编码，Excel 可直接打开不乱码；XLSX 含「保质期清单」与「汇总」两个工作表，临期行黄色、过期行红色高亮。
      </div>
    </div>
  )
}
