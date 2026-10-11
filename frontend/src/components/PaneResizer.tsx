import { useRef, useState } from 'react';

interface Props {
  min: number;
  max: number;
  onChange: (width: number) => void;
  onReset: () => void;
}

const KEY_STEP = 24;

/** Drag handle on the left edge of a right-hand pane. */
export function PaneResizer({ min, max, onChange, onReset }: Props) {
  const [dragging, setDragging] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const clamp = (w: number) => Math.round(Math.min(Math.max(w, min), max));

  const onPointerMove = (e: React.PointerEvent) => {
    if (!dragging) return;
    const pane = ref.current?.parentElement;
    if (!pane) return;
    onChange(clamp(pane.getBoundingClientRect().right - e.clientX));
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    const width = ref.current?.parentElement?.getBoundingClientRect().width ?? min;
    if (e.key === 'ArrowLeft') onChange(clamp(width + KEY_STEP));
    else if (e.key === 'ArrowRight') onChange(clamp(width - KEY_STEP));
    else if (e.key === 'Home') onReset();
    else return;
    e.preventDefault();
  };

  return (
    <div
      ref={ref}
      className={`resizer${dragging ? ' dragging' : ''}`}
      role="separator"
      aria-orientation="vertical"
      aria-label="Resize source viewer"
      aria-valuemin={min}
      aria-valuemax={max}
      tabIndex={0}
      title="Drag to resize · double-click to reset"
      onPointerDown={(e) => {
        e.preventDefault();
        e.currentTarget.setPointerCapture(e.pointerId);
        setDragging(true);
      }}
      onPointerMove={onPointerMove}
      onPointerUp={() => setDragging(false)}
      onPointerCancel={() => setDragging(false)}
      onDoubleClick={onReset}
      onKeyDown={onKeyDown}
    />
  );
}
