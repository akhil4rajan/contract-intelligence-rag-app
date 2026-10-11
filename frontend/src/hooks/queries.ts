import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '../api/client';

export const keys = {
  contracts: ['contracts'] as const,
  conversations: ['conversations'] as const,
  conversation: (id: string) => ['conversation', id] as const,
  health: ['health'] as const,
  page: (id: string, n: number) => ['page', id, n] as const,
  extraction: (id: string) => ['extraction', id] as const,
};

export function useContracts() {
  return useQuery({
    queryKey: keys.contracts,
    queryFn: api.listContracts,
    // Poll while any contract is still being indexed.
    refetchInterval: (q) => (q.state.data?.some((c) => c.status === 'processing') ? 1500 : false),
  });
}

export function useUploadContracts() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (files: File[]) => api.uploadContracts(files),
    onSettled: () => qc.invalidateQueries({ queryKey: keys.contracts }),
  });
}

export function useDeleteContract() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.deleteContract(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.contracts }),
  });
}

export function useConversations() {
  return useQuery({ queryKey: keys.conversations, queryFn: api.listConversations });
}

export function useConversation(id: string | null) {
  return useQuery({
    queryKey: keys.conversation(id ?? ''),
    queryFn: () => api.getConversation(id as string),
    enabled: !!id,
  });
}

export function useDeleteConversation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.deleteConversation(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.conversations }),
  });
}

export function useAsk() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.query,
    // Awaited so the new message is in the cache before the pending bubble is cleared.
    onSuccess: async (data) => {
      await Promise.all([
        qc.invalidateQueries({ queryKey: keys.conversations }),
        qc.invalidateQueries({ queryKey: keys.conversation(data.conversation_id) }),
      ]);
    },
  });
}

export function useHealth() {
  return useQuery({ queryKey: keys.health, queryFn: api.health, staleTime: 60_000 });
}

export function usePage(contractId: string | null, page: number, enabled: boolean) {
  return useQuery({
    queryKey: keys.page(contractId ?? '', page),
    queryFn: () => api.getPage(contractId as string, page),
    enabled: enabled && !!contractId,
  });
}

export function useExtraction(contractId: string) {
  return useQuery({
    queryKey: keys.extraction(contractId),
    queryFn: () => api.getExtraction(contractId),
    // The backend caches the result, so there is no reason to refetch on focus.
    staleTime: 5 * 60_000,
    retry: false,
  });
}

export function useRefreshExtraction(contractId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.refreshExtraction(contractId),
    onSuccess: (data) => qc.setQueryData(keys.extraction(contractId), data),
  });
}
