import { useState } from 'react'
import { toast } from 'sonner'
import { Pencil, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { SubjectPicker } from '@/components/search/SubjectPicker'
import { updateProfile, deleteProfile } from '@/lib/api'
import type { UserProfile } from '@/types'

interface UserCardProps {
  profile: UserProfile
  onUpdate: () => void
}

function avatarLetters(username: string): string {
  return username.slice(0, 2).toUpperCase()
}

export function UserCard({ profile, onUpdate }: UserCardProps) {
  const [editing, setEditing] = useState(false)
  const [username, setUsername] = useState(profile.username)
  const [email, setEmail] = useState(profile.email)
  const [subjects, setSubjects] = useState<string[]>(profile.favorite_subjects)
  const [saving, setSaving] = useState(false)

  async function handleSave() {
    setSaving(true)
    try {
      await updateProfile({ username, email, favorite_subjects: subjects })
      toast.success('Profile updated')
      setEditing(false)
      onUpdate()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Failed to update profile')
    } finally {
      setSaving(false)
    }
  }

  function handleCancel() {
    setUsername(profile.username)
    setEmail(profile.email)
    setSubjects(profile.favorite_subjects)
    setEditing(false)
  }

  async function handleDelete() {
    const input = prompt('Type DELETE to permanently remove your account (this cannot be undone).')
    if (input !== 'DELETE') return
    try {
      await deleteProfile()
      window.location.href = '/logout'
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Failed to delete account')
    }
  }

  return (
    <div className="rounded-xl border border-border bg-card p-8">
      {!editing ? (
        <div>
          {/* Top row: avatar + name/email + edit button */}
          <div className="flex items-center gap-5 mb-6">
            <div className="flex h-20 w-20 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground text-2xl font-bold">
              {avatarLetters(profile.username)}
            </div>
            <div className="flex-1 min-w-0">
              <h1 className="text-2xl font-bold leading-tight truncate">{profile.username}</h1>
              <p className="text-sm text-muted-foreground truncate">{profile.email}</p>
            </div>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setEditing(true)}
              className="shrink-0 self-start"
            >
              <Pencil className="h-3.5 w-3.5 sm:mr-1.5" />
              <span className="hidden sm:inline">Edit</span>
            </Button>
          </div>

          {/* Stats row */}
          <div className="flex gap-4 mb-6">
            {[
              { value: profile.num_books_read, label: 'books read' },
              { value: profile.num_ratings, label: 'ratings' },
            ].map(({ value, label }) => (
              <div
                key={label}
                className="flex-1 rounded-lg bg-muted/50 px-4 py-3 text-center"
              >
                <p className="text-2xl font-bold">{value.toLocaleString()}</p>
                <p className="text-xs text-muted-foreground mt-0.5">{label}</p>
              </div>
            ))}
          </div>

          {/* Favorite subjects */}
          {profile.favorite_subjects.length > 0 && (
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">
                Favourite Subjects
              </p>
              <div className="flex flex-wrap gap-2">
                {profile.favorite_subjects.map((s) => (
                  <span
                    key={s}
                    className="inline-block rounded-full bg-secondary px-3 py-1 text-xs font-medium text-secondary-foreground"
                  >
                    {s}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        <div>
          <div className="flex items-center justify-between mb-5">
            <h2 className="text-base font-semibold">Edit Profile</h2>
            <button
              type="button"
              onClick={handleCancel}
              className="text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          <div className="space-y-4 max-w-md">
            <div>
              <label className="block text-sm font-medium mb-1" htmlFor="edit-username">
                Username
              </label>
              <input
                id="edit-username"
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1" htmlFor="edit-email">
                Email
              </label>
              <input
                id="edit-email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">
                Favourite Subjects <span className="text-muted-foreground font-normal">(up to 5)</span>
              </label>
              <SubjectPicker selected={subjects} onChange={setSubjects} />
            </div>

            <div className="flex items-center gap-2 pt-1">
              <Button type="button" size="sm" onClick={handleSave} disabled={saving}>
                {saving ? 'Saving…' : 'Save'}
              </Button>
              <Button type="button" variant="outline" size="sm" onClick={handleCancel}>
                Cancel
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={handleDelete}
                className="ml-auto text-destructive border-destructive/40 hover:bg-destructive/10 hover:text-destructive"
              >
                Delete Account
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
