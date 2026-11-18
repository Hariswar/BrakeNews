import { createRoot } from "react-dom/client";
import VehicleExplorer, { type ExplorerProps } from "./VehicleExplorer";

export function mount(element: HTMLElement, props: ExplorerProps) {
  const root = createRoot(element);
  root.render(<VehicleExplorer {...props} />);
  return () => root.unmount();
}
