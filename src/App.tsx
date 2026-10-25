import { Routes, Route, Navigate } from 'react-router-dom'
import AppLayout from '@/components/AppLayout'
import Dashboard from '@/pages/Dashboard'
import ImportPage from '@/pages/ImportPage'
import ItemsPage from '@/pages/ItemsPage'
import ExportPage from '@/pages/ExportPage'
import SettingsPage from '@/pages/SettingsPage'

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/import" element={<ImportPage />} />
        <Route path="/items" element={<ItemsPage />} />
        <Route path="/export" element={<ExportPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
