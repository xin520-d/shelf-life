import { NavLink, Outlet } from 'react-router-dom'
import {
  LayoutDashboard, Camera, Package, Download, Settings, Refrigerator,
  CalendarDays, Cpu,
} from 'lucide-react'
import { MetaProvider } from '@/hooks/useMeta'

const NAV = [
  { to: '/', label: '仪表盘', icon: LayoutDashboard, desc: '临期总览' },
  { to: '/import', label: '拍照导入', icon: Camera, desc: '识别 · 归类 · 日期' },
  { to: '/items', label: '物品清单', icon: Package, desc: '在册物品管理' },
  { to: '/export', label: '数据导出', icon: Download, desc: '表格文件' },
  { to: '/settings', label: '设置', icon: Settings, desc: '类别 · 阈值' },
]

function todayText() {
  const d = new Date()
  const week = ['日', '一', '二', '三', '四', '五', '六'][d.getDay()]
  return `${d.getFullYear()} 年 ${d.getMonth() + 1} 月 ${d.getDate()} 日 · 星期${week}`
}

export default function AppLayout() {
  return (
    <MetaProvider>
      <div className="flex min-h-screen bg-slate-50 text-slate-800">
        {/* 侧边栏 */}
        <aside className="fixed inset-y-0 left-0 z-30 flex w-60 flex-col border-r border-slate-200 bg-white">
          <div className="flex items-center gap-3 border-b border-slate-100 px-5 py-5">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-sky-500 to-indigo-600 text-white shadow-sm">
              <Refrigerator className="h-5 w-5" />
            </div>
            <div>
              <div className="text-[15px] font-bold leading-tight">保质期管家</div>
              <div className="text-xs text-slate-400">拍照识别 · 临期提醒</div>
            </div>
          </div>

          <nav className="flex-1 space-y-1 px-3 py-4">
            {NAV.map(({ to, label, icon: Icon, desc }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  `group flex items-start gap-3 rounded-lg px-3 py-2.5 transition-colors ${
                    isActive
                      ? 'bg-sky-50 text-sky-700'
                      : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
                  }`
                }
              >
                {({ isActive }) => (
                  <>
                    <Icon className={`mt-0.5 h-[18px] w-[18px] ${isActive ? 'text-sky-600' : 'text-slate-400 group-hover:text-slate-600'}`} />
                    <div>
                      <div className="text-sm font-medium leading-tight">{label}</div>
                      <div className="text-[11px] text-slate-400">{desc}</div>
                    </div>
                  </>
                )}
              </NavLink>
            ))}
          </nav>

          <div className="border-t border-slate-100 px-4 py-3">
            <div className="flex items-center gap-2 text-[11px] text-slate-400">
              <Cpu className="h-3.5 w-3.5" />
              <span>YOLO11n · RapidOCR · 本地推理</span>
            </div>
          </div>
        </aside>

        {/* 主区域 */}
        <div className="ml-60 flex min-h-screen flex-1 flex-col">
          <header className="sticky top-0 z-20 flex h-14 items-center justify-between border-b border-slate-200 bg-white/80 px-8 backdrop-blur">
            <div className="flex items-center gap-2 text-sm text-slate-500">
              <CalendarDays className="h-4 w-4" />
              {todayText()}
            </div>
            <div className="flex items-center gap-2">
              <span className="rounded-full bg-emerald-50 px-2.5 py-1 text-xs font-medium text-emerald-700">
                模型服务就绪
              </span>
            </div>
          </header>

          <main className="mx-auto w-full max-w-7xl flex-1 px-8 py-6">
            <Outlet />
          </main>
        </div>
      </div>
    </MetaProvider>
  )
}
