package bot

func (c *Client) resetDemoConnection() {
	if c.demo == nil {
		return
	}
	c.decoder.CaptureDemo = true
	if err := c.demo.Reset(); err != nil {
		c.demoErr = err
	}
}
