// Shared TypeScript types for the app
export type Message = {
  id: string
  text: string
  role?: 'user' | 'assistant' | 'system'
}
