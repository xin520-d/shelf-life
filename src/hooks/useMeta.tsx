import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import { api, type Category } from '@/lib/api'

interface MetaValue {
  categories: Category[]
  loading: boolean
  reload: () => Promise<void>
}

const MetaContext = createContext<MetaValue>({
  categories: [], loading: true, reload: async () => {},
})

export function useMeta() {
  return useContext(MetaContext)
}

export function MetaProvider({ children }: { children: ReactNode }) {
  const [categories, setCategories] = useState<Category[]>([])
  const [loading, setLoading] = useState(true)

  const load = async () => {
    try {
      setCategories(await api.categories())
    } catch (e) {
      console.error('加载元数据失败', e)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  return (
    <MetaContext.Provider value={{ categories, loading, reload: load }}>
      {children}
    </MetaContext.Provider>
  )
}
