package quake

// InventoryItem is authoritative svc_inventory data, named by CS_ITEMS.
type InventoryItem struct {
	ID    int    `json:"id"`
	Name  string `json:"name"`
	Count int    `json:"count"`
}
