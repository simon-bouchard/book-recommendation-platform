import { useState, useEffect } from 'react'
import { fetchProfile } from '@/lib/api'
import type { UserProfile } from '@/types'
import { UserCard } from './UserCard'
import { InteractionsList } from './InteractionsList'
import { ProfileRecommendations } from './ProfileRecommendations'
import { Skeleton } from '@/components/ui/skeleton'

function ProfileSkeleton() {
  return (
    <div className="mx-auto max-w-5xl px-6 pt-10 pb-16 space-y-8">
      <Skeleton className="h-32 w-full rounded-xl" />
      <Skeleton className="h-64 w-full rounded-xl" />
    </div>
  )
}

export function ProfilePage() {
  const [profile, setProfile] = useState<UserProfile | null>(null)
  const [error, setError] = useState<string | null>(null)

  function reload() {
    fetchProfile()
      .then(setProfile)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : 'Failed to load profile'))
  }

  useEffect(() => { reload() }, [])

  if (error) {
    return (
      <div className="mx-auto max-w-5xl px-6 pt-16 text-center text-muted-foreground">
        {error}
      </div>
    )
  }

  if (!profile) return <ProfileSkeleton />

  return (
    <div className="mx-auto max-w-5xl px-6 pt-10 pb-16 space-y-10">
      <UserCard profile={profile} onUpdate={reload} />

      <InteractionsList />

      <div className="border-t border-border pt-10">
        <ProfileRecommendations userId={profile.id} numRatings={profile.num_ratings} />
      </div>
    </div>
  )
}
