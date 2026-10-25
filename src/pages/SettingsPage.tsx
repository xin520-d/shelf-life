import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import { Plus, Trash2, Pencil, Check, X, BrainCircuit, Download, Loader2, ScanLine, FileText, Eye } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from '@/components/ui/table'
import { api, type Category, type Correction, type ModelInfo } from '@/lib/api'
import { useMeta } from '@/hooks/useMeta'

const PRESET_COLORS = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6', '#ec4899', '#6b7280', '#14b8a6']

const MODEL_ICONS: Record<string, typeof ScanLine> = {
  yolo: ScanLine, ocr: FileText, clip: Eye,
}

export default function SettingsPage() {
  const { categories, reload } = useMeta()
  const [nearDays, setNearDays] = useState(7)
  const [lowConf, setLowConf] = useState(0.6)
  const [savingDays, setSavingDays] = useState(false)
  const [corrections, setCorrections] = useState<Correction[]>([])

  // 新类别表单
  const [newCat, setNewCat] = useState({ name: '', color: '#3b82f6', days: 365 })
  // 编辑中的类别
  const [editCatId, setEditCatId] = useState<number | null>(null)
  const [editCat, setEditCat] = useState({ name: '', color: '#3b82f6', days: 365 })
  // 识别模型状态
  const [models, setModels] = useState<Record<string, ModelInfo> | null>(null)
  const [clipInstalling, setClipInstalling] = useState(false)

  useEffect(() => {
    api.settings().then(s => { setNearDays(s.near_expiry_days); setLowConf(s.low_conf) })
    api.corrections().then(setCorrections)
    api.models().then(setModels)
  }, [])

  const installClip = async () => {
    setClipInstalling(true)
    try {
      await api.downloadClip()
      toast.success('已开始下载 CLIP 模型（约 1.7GB），下载期间不影响其他识别通道')
      // 轮询等待安装完成
      const timer = setInterval(async () => {
        const m = await api.models()
        setModels(m)
        if (m.clip?.ready || m.clip?.download_state === 'failed') {
          clearInterval(timer)
          setClipInstalling(false)
          if (m.clip?.ready) toast.success('CLIP 视觉识别模型安装完成')
          else toast.error(`安装失败：${m.clip?.download_error || '未知错误'}`)
        }
      }, 3000)
    } catch (e) {
      setClipInstalling(false)
      toast.error((e as Error).message)
    }
  }

  const saveSettings = async () => {
    setSavingDays(true)
    try {
      await api.updateSettings({ near_expiry_days: nearDays, low_conf: lowConf })
      toast.success('设置已保存')
    } finally {
      setSavingDays(false)
    }
  }

  const addCategory = async () => {
    if (!newCat.name.trim()) return toast.error('请填写类别名称')
    try {
      await api.createCategory({ name: newCat.name.trim(), color: newCat.color, default_shelf_life_days: newCat.days })
      setNewCat({ name: '', color: '#3b82f6', days: 365 })
      await reload()
      toast.success('类别已添加')
    } catch (e) { toast.error((e as Error).message) }
  }

  const saveCat = async (id: number) => {
    await api.updateCategory(id, { name: editCat.name.trim(), color: editCat.color, default_shelf_life_days: editCat.days })
    setEditCatId(null)
    await reload()
    toast.success('类别已更新')
  }

  const delCategory = async (c: Category) => {
    if (c.is_builtin) return toast.error('内置类别不可删除')
    if (!confirm(`删除类别「${c.name}」？其中物品将移到「其他」。`)) return
    await api.deleteCategory(c.id)
    await reload()
    toast.success('类别已删除')
  }

  return (
    <div className="max-w-4xl space-y-5">
      <div>
        <h1 className="text-lg font-bold">设置</h1>
        <p className="mt-1 text-sm text-slate-500">临期阈值、类别体系与识别修正记忆</p>
      </div>

      {/* 识别与提醒阈值 */}
      <Card>
        <CardHeader><CardTitle className="text-base">识别与提醒阈值</CardTitle></CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-end gap-3">
            <div>
              <Label className="text-xs text-slate-400">距离到期 ≤ N 天时标记为「临期」</Label>
              <Input type="number" min={1} max={90} value={nearDays}
                onChange={e => setNearDays(Number(e.target.value))} className="mt-1 w-24" />
            </div>
            <span className="pb-2 text-xs text-slate-400">影响看板、列表状态徽章与导出着色</span>
          </div>
          <div className="flex flex-wrap items-end gap-3">
            <div>
              <Label className="text-xs text-slate-400">识别置信度低于此值时标记为「待确认」</Label>
              <Input type="number" min={0} max={1} step={0.05} value={lowConf}
                onChange={e => setLowConf(Math.min(1, Math.max(0, Number(e.target.value))))}
                className="mt-1 w-24" />
            </div>
            <span className="pb-2 text-xs text-slate-400">值越低越宽松（少弹审核）；值越高越严格（更多待确认）</span>
          </div>
          <Button onClick={saveSettings} disabled={savingDays} className="bg-sky-600 hover:bg-sky-700">
            {savingDays ? '保存中…' : '保存设置'}
          </Button>
        </CardContent>
      </Card>

      {/* 类别体系 */}
      <Card>
        <CardHeader><CardTitle className="text-base">类别体系（物品自动归类的目标类别）</CardTitle></CardHeader>
        <CardContent className="space-y-3">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>颜色</TableHead><TableHead>名称</TableHead>
                <TableHead>默认保质期（天）</TableHead><TableHead className="text-center">在册数量</TableHead>
                <TableHead className="text-right">操作</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {categories.map(c => (
                <TableRow key={c.id}>
                  {editCatId === c.id ? (
                    <>
                      <TableCell>
                        <div className="flex gap-1">
                          {PRESET_COLORS.slice(0, 5).map(col => (
                            <button key={col} onClick={() => setEditCat({ ...editCat, color: col })}
                              className={`h-4 w-4 rounded-full ring-1 ring-slate-200 ${editCat.color === col ? 'outline outline-2 outline-offset-1 outline-slate-400' : ''}`}
                              style={{ background: col }} />
                          ))}
                        </div>
                      </TableCell>
                      <TableCell><Input value={editCat.name} className="h-7 w-28 text-sm"
                        onChange={e => setEditCat({ ...editCat, name: e.target.value })} /></TableCell>
                      <TableCell><Input type="number" value={editCat.days} className="h-7 w-20 text-sm"
                        onChange={e => setEditCat({ ...editCat, days: Number(e.target.value) })} /></TableCell>
                      <TableCell className="text-center text-sm">{c.item_count}</TableCell>
                      <TableCell className="text-right">
                        <Button size="icon" variant="ghost" className="h-7 w-7 text-emerald-600" onClick={() => saveCat(c.id)}><Check className="h-3.5 w-3.5" /></Button>
                        <Button size="icon" variant="ghost" className="h-7 w-7" onClick={() => setEditCatId(null)}><X className="h-3.5 w-3.5" /></Button>
                      </TableCell>
                    </>
                  ) : (
                    <>
                      <TableCell><span className="block h-4 w-4 rounded-full" style={{ background: c.color }} /></TableCell>
                      <TableCell className="font-medium">
                        {c.name}
                        {c.is_builtin && <Badge variant="outline" className="ml-2 text-[10px] text-slate-400">内置</Badge>}
                      </TableCell>
                      <TableCell className="text-sm tabular-nums">{c.default_shelf_life_days}</TableCell>
                      <TableCell className="text-center text-sm">{c.item_count}</TableCell>
                      <TableCell className="text-right">
                        <Button size="icon" variant="ghost" className="h-7 w-7"
                          onClick={() => { setEditCatId(c.id); setEditCat({ name: c.name, color: c.color, days: c.default_shelf_life_days }) }}>
                          <Pencil className="h-3.5 w-3.5" />
                        </Button>
                        <Button size="icon" variant="ghost" className="h-7 w-7 text-red-500" onClick={() => delCategory(c)}>
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      </TableCell>
                    </>
                  )}
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <div className="flex flex-wrap items-end gap-2 rounded-lg bg-slate-50 p-3">
            <div>
              <Label className="text-[11px] text-slate-400">名称</Label>
              <Input value={newCat.name} placeholder="如：乳制品" className="mt-1 h-8 w-28 text-sm"
                onChange={e => setNewCat({ ...newCat, name: e.target.value })} />
            </div>
            <div>
              <Label className="text-[11px] text-slate-400">颜色</Label>
              <div className="mt-1.5 flex gap-1">
                {PRESET_COLORS.map(col => (
                  <button key={col} onClick={() => setNewCat({ ...newCat, color: col })}
                    className={`h-5 w-5 rounded-full ring-1 ring-slate-200 ${newCat.color === col ? 'outline outline-2 outline-offset-1 outline-slate-400' : ''}`}
                    style={{ background: col }} />
                ))}
              </div>
            </div>
            <div>
              <Label className="text-[11px] text-slate-400">默认保质期（天）</Label>
              <Input type="number" value={newCat.days} className="mt-1 h-8 w-20 text-sm"
                onChange={e => setNewCat({ ...newCat, days: Number(e.target.value) })} />
            </div>
            <Button size="sm" onClick={addCategory}><Plus className="mr-1 h-3.5 w-3.5" />添加类别</Button>
          </div>
        </CardContent>
      </Card>

      {/* 识别模型 */}
      <Card>
        <CardHeader><CardTitle className="text-base">识别模型（三条互补通道）</CardTitle></CardHeader>
        <CardContent className="space-y-2">
          <p className="text-xs text-slate-500">
            识别走「YOLO 视觉检测 + RapidOCR 包装文字 + CLIP 零样本图文匹配」三条通道，互相印证：
            YOLO 认得出香蕉瓶子等通用外形，OCR 读包装上的品名与日期，CLIP 在前两者拿不准时给出视觉初判。
          </p>
          {models && (['yolo', 'ocr', 'clip'] as const).map(key => {
            const m = models[key]
            const Icon = MODEL_ICONS[key] || ScanLine
            return (
              <div key={key} className="flex items-center gap-3 rounded-lg border border-slate-100 px-3 py-2">
                <Icon className="h-4 w-4 shrink-0 text-slate-400" />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium">{m.name}</span>
                    <Badge variant="outline"
                      className={m.ready ? 'border-emerald-200 text-emerald-600' : 'border-amber-200 text-amber-600'}>
                      {m.ready ? '已就绪' : (m.download_state === 'downloading' ? '下载中…' : '未安装')}
                    </Badge>
                  </div>
                  <p className="truncate text-xs text-slate-400">{m.desc}</p>
                </div>
                {key === 'clip' && !m.ready && (
                  <Button size="sm" variant="outline" disabled={clipInstalling} onClick={installClip}>
                    {clipInstalling
                      ? <><Loader2 className="mr-1 h-3.5 w-3.5 animate-spin" />安装中</>
                      : <><Download className="mr-1 h-3.5 w-3.5" />一键安装</>}
                  </Button>
                )}
              </div>
            )
          })}
        </CardContent>
      </Card>

      {/* 修正记忆 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <BrainCircuit className="h-4 w-4 text-violet-500" />
            识别修正记忆
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="mb-3 text-xs text-slate-500">
            当你把某个模型识别结果（如 bottle）改到其他类别时，系统会记住这个选择；下次识别到同样的原始类别名，将优先按你的习惯归类。
          </p>
          {corrections.length === 0 ? (
            <div className="rounded-lg bg-slate-50 py-8 text-center text-sm text-slate-400">
              暂无修正记录——在清单里修改一次物品类别/名称后即会出现在这里
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>模型原始输出</TableHead><TableHead>常用名称</TableHead>
                  <TableHead>归类到</TableHead><TableHead className="text-center">沿用次数</TableHead>
                  <TableHead className="text-right">操作</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {corrections.map(m => (
                  <TableRow key={m.id}>
                    <TableCell className="font-mono text-xs">{m.raw_name}</TableCell>
                    <TableCell className="text-sm">{m.item_name || '—'}</TableCell>
                    <TableCell className="text-sm">{m.category_name}</TableCell>
                    <TableCell className="text-center text-sm tabular-nums">{m.use_count}</TableCell>
                    <TableCell className="text-right">
                      <Button size="icon" variant="ghost" className="h-7 w-7 text-red-500"
                        onClick={async () => {
                          await api.deleteCorrection(m.id)
                          setCorrections(await api.corrections())
                          toast.success('已删除该记忆')
                        }}>
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
