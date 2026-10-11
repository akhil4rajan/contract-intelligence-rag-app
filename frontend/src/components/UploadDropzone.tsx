import { useRef, useState } from 'react';
import { useUploadContracts } from '../hooks/queries';
import { MAX_UPLOAD_BYTES, formatBytes, isPdf } from '../utils';

interface Notice {
  kind: 'ok' | 'warn' | 'error';
  text: string;
}

export function UploadDropzone() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [notices, setNotices] = useState<Notice[]>([]);
  const upload = useUploadContracts();

  const submit = (list: FileList | File[]) => {
    const files = Array.from(list);
    const local: Notice[] = [];
    const valid = files.filter((f) => {
      if (!isPdf(f)) {
        local.push({ kind: 'error', text: `${f.name}: only PDF files are supported` });
        return false;
      }
      if (f.size > MAX_UPLOAD_BYTES) {
        local.push({ kind: 'error', text: `${f.name}: exceeds ${formatBytes(MAX_UPLOAD_BYTES)}` });
        return false;
      }
      return true;
    });
    setNotices(local);
    if (!valid.length) return;
    upload.mutate(valid, {
      onSuccess: (results) => {
        setNotices([
          ...local,
          ...results.map<Notice>((r) =>
            r.status === 'accepted'
              ? { kind: 'ok', text: `${r.filename}: uploaded, indexing…` }
              : r.status === 'duplicate'
                ? { kind: 'warn', text: `${r.filename}: already uploaded` }
                : { kind: 'error', text: `${r.filename}: ${r.error?.message ?? 'rejected'}` },
          ),
        ]);
      },
      onError: (err) => setNotices([...local, { kind: 'error', text: err.message }]),
    });
  };

  return (
    <div className="upload">
      <div
        className={`dropzone${dragging ? ' dragging' : ''}`}
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (e.dataTransfer.files.length) submit(e.dataTransfer.files);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          multiple
          hidden
          onChange={(e) => {
            if (e.target.files?.length) submit(e.target.files);
            e.target.value = '';
          }}
        />
        {upload.isPending ? (
          <span>Uploading…</span>
        ) : (
          <>
            <strong>Upload contracts</strong>
            <span className="muted">Drop PDFs here or click to browse</span>
          </>
        )}
      </div>
      {notices.length > 0 && (
        <ul className="notices">
          {notices.map((n, i) => (
            <li key={i} className={`notice ${n.kind}`}>
              {n.text}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
