import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Package, AlertTriangle, CalendarX2, Clock, ChevronRight, Camera,
} from 'lucide-react'
import {
  PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  Tooltip, CartesianGrid,
} from 'recharts'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { api, type Item, type Stats, STATUS_STYLE, STATUS_LABEL, remainingText } from '@/lib/api'

function StatCard({ icon: Icon, label, value, sub, tone, to }: {
  icon: typeof Package; label: string; value: number; sub?: string
  tone: 'sky' | 'amber' | 'red' | 'emerald'; to?: string
}) {
  const tones = {
    sky: 'from-sky-500 to-sky-600 shadow-sky-200',
    amber: 'from-amber-500 to-orange-500 shadow-amber-200',
    red: 'from-rose-500 to-red-600 shadow-rose-200',
    emerald: 'from-emerald-500 to-teal-600 shadow-emerald-200',
  }
  const card = (
    <Card className={`overflow-hidden ${to ? 'transition hover:shadow-md' : ''}`}>
      <CardContent className="flex items-center gap-4 p-5">
        <div className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br text-white shadow-lg ${tones[tone]}`}>
          <Icon className="h-6 w-6" />
        </div>
        <div className="min-w-0">
          <div className="text-2xl font-bold leading-tight tabular-nums">{value}</div>
          <div className="truncate text-sm text-slate-500">{label}</div>
          {sub && <div className="mt-0.5 text-[11px] text-slate-400">{sub}</div>}
        </div>
      </CardContent>
    </Card>
  )
  return to ? <Link to={to}>{card}</Link> : card
}

export default function Dashboard() {
  const [stats, setStats] = useState<Stats | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    api.stats().then(setStats).catch(e => setError(e.message))
  }, [])

  if (error) {
    return <div className="rounded-lg border border-red-200 bg-red-50 p-6 text-red-700">加载失败：{error}</div>
  }
  if (!stats) {
    return <div className="py-20 text-center text-slate-400">正在加载看板数据…</div>
  }

  const timeline = stats.expiry_timeline
    .filter((_, i) => i % 3 === 0 || i < 8) // 稀疏展示，保留近 8 天粒度
    .map(t => ({ ...t, label: t.date.slice(5).replace('-', '/') }))
  const pieData = stats.by_category.map(c => ({ name: c.name, value: c.count, color: c.color }))

  return (
    <div className="space-y-6">
      {/* 欢迎条 */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-sky-100 bg-gradient-to-r from-sky-50 via-white to-indigo-50 p-5">
        <div>
          <h1 className="text-lg font-bold">临期看板</h1>
          <p className="mt-1 text-sm text-slate-500">
            拍照导入 → 识别归类 → 日期抽取 → 临期分档，全流程数据总览
          </p>
        </div>
        <Button asChild className="bg-sky-600 hover:bg-sky-700">
          <Link to="/import"><Camera className="mr-2 h-4 w-4" />拍照导入新物品</Link>
        </Button>
      </div>

      {/* 统计卡片 */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard icon={Package} label="在册物品" value={stats.total_active}
          sub={`另有 ${stats.unknown} 件日期未知`} tone="sky" to="/items?tab=active" />
        <StatCard icon={AlertTriangle} label="临期预警" value={stats.near_expiry}
          sub="即将到期，优先消耗" tone="amber" to="/items?tab=near" />
        <StatCard icon={CalendarX2} label="已过期" value={stats.expired}
          sub="建议及时清理" tone="red" to="/items?tab=expired" />
        <StatCard icon={Clock} label="7 天内到期" value={stats.due_7d}
          sub={`30 天内共 ${stats.due_30d} 件`} tone="emerald" to="/items?tab=near" />
      </div>

      {/* 图表区 */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-5">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="text-base">类别分布</CardTitle>
          </CardHeader>
          <CardContent>
            {pieData.length > 0 ? (
              <div className="flex items-center gap-4">
                <ResponsiveContainer width={180} height={180}>
                  <PieChart>
                    <Pie data={pieData} dataKey="value" nameKey="name"
                      innerRadius={50} outerRadius={80} paddingAngle={3} stroke="none">
                      {pieData.map((d, i) => <Cell key={i} fill={d.color} />)}
                    </Pie>
                    <Tooltip formatter={(v: number, n: string) => [`${v} 件`, n]} />
                  </PieChart>
                </ResponsiveContainer>
                <div className="flex-1 space-y-2">
                  {pieData.map(d => (
                    <div key={d.name} className="flex items-center justify-between text-sm">
                      <span className="flex items-center gap-2">
                        <span className="h-2.5 w-2.5 rounded-full" style={{ background: d.color }} />
                        {d.name}
                      </span>
                      <span className="tabular-nums text-slate-500">{d.value} 件</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="py-10 text-center text-sm text-slate-400">暂无数据</div>
            )}
          </CardContent>
        </Card>

        <Card className="lg:col-span-3">
          <CardHeader>
            <CardTitle className="text-base">未来 30 天到期趋势</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={timeline} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                <XAxis dataKey="label" tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} allowDecimals={false} />
                <Tooltip
                  contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 12 }}
                  formatter={(v: number) => [`${v} 件`, '到期数量']}
                  labelFormatter={(l: string) => `日期 ${l}`}
                />
                <Bar dataKey="count" fill="#0ea5e9" radius={[4, 4, 0, 0]} maxBarSize={28} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* 临期优先清单 */}
      <Card>
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <CardTitle className="text-base">最紧急的物品</CardTitle>
          <Button variant="ghost" size="sm" asChild>
            <Link to="/items">查看全部 <ChevronRight className="ml-1 h-4 w-4" /></Link>
          </Button>
        </CardHeader>
        <CardContent className="p-0">
          {stats.recent_items.length === 0 ? (
            <div className="px-6 pb-8 pt-2 text-center text-sm text-slate-400">
              暂无在册物品，去<Link to="/import" className="mx-1 text-sky-600 underline">拍照导入</Link>吧
            </div>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-100 text-left text-xs text-slate-400">
                  <th className="px-6 py-2 font-medium">物品</th>
                  <th className="px-4 py-2 font-medium">类别</th>
                  <th className="px-4 py-2 font-medium">到期日期</th>
                  <th className="px-4 py-2 font-medium">剩余</th>
                  <th className="px-6 py-2 font-medium">状态</th>
                </tr>
              </thead>
              <tbody>
                {stats.recent_items.map((it: Item) => (
                  <tr key={it.id} className="border-b border-slate-50 last:border-0 hover:bg-slate-50/60">
                    <td className="px-6 py-2.5 font-medium">{it.name}</td>
                    <td className="px-4 py-2.5">
                      <span className="flex items-center gap-1.5">
                        <span className="h-2 w-2 rounded-full" style={{ background: it.category_color }} />
                        {it.category_name}
                      </span>
                    </td>
                    <td className="px-4 py-2.5 tabular-nums text-slate-500">{it.expiry_date ?? '—'}</td>
                    <td className="px-4 py-2.5 tabular-nums">{remainingText(it.remaining_days)}</td>
                    <td className="px-6 py-2.5">
                      <Badge variant="outline" className={STATUS_STYLE[it.expiry_status]}>
                        {STATUS_LABEL[it.expiry_status]}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
