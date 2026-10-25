import { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props {
  children: ReactNode
}

interface State {
  error: Error | null
}

/**
 * 全局错误边界：React 渲染崩溃时整个树会卸载变成白屏，
 * 这里兜底显示友好提示 + 刷新按钮，避免用户看到空白页。
 */
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('页面渲染崩溃:', error, info.componentStack)
  }

  render() {
    if (this.state.error) {
      return (
        <div className="flex min-h-screen items-center justify-center bg-slate-50 p-8">
          <div className="max-w-md rounded-xl border border-amber-200 bg-white p-6 text-center shadow-sm">
            <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-amber-50 text-2xl">
              ⚠️
            </div>
            <h1 className="mb-2 text-lg font-bold text-slate-800">页面出错了</h1>
            <p className="mb-4 text-sm text-slate-500">
              界面渲染时发生异常，你的数据不会丢失。刷新页面即可恢复。
            </p>
            <pre className="mb-4 max-h-24 overflow-auto rounded-lg bg-slate-50 p-3 text-left text-[11px] text-slate-400">
              {this.state.error.message}
            </pre>
            <button
              onClick={() => window.location.reload()}
              className="rounded-lg bg-sky-600 px-5 py-2 text-sm font-medium text-white hover:bg-sky-700"
            >
              刷新页面
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}
