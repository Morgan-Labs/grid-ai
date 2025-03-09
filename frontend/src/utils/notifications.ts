type NotificationColor = 'blue' | 'green' | 'red' | 'yellow' | 'gray' | 'orange';

interface NotificationProps {
  title?: string;
  message: string;
  color?: NotificationColor;
  autoClose?: number;
}

// Simple notification system using DOM API
class NotificationSystem {
  private getContainer(): HTMLElement {
    let container = document.getElementById('notifications-container');
    
    if (!container) {
      container = document.createElement('div');
      container.id = 'notifications-container';
      container.style.position = 'fixed';
      container.style.top = '20px';
      container.style.right = '20px';
      container.style.zIndex = '9999';
      container.style.display = 'flex';
      container.style.flexDirection = 'column';
      container.style.gap = '10px';
      document.body.appendChild(container);
    }
    
    return container;
  }
  
  show({ title, message, color = 'blue', autoClose = 5000 }: NotificationProps): void {
    const container = this.getContainer();
    
    // Create notification element
    const notification = document.createElement('div');
    notification.style.backgroundColor = `var(--mantine-color-${color}-light)`;
    notification.style.borderLeft = `4px solid var(--mantine-color-${color}-filled)`;
    notification.style.borderRadius = 'var(--mantine-radius-sm)';
    notification.style.boxShadow = 'var(--mantine-shadow-md)';
    notification.style.padding = '16px';
    notification.style.maxWidth = '400px';
    notification.style.width = '100%';
    notification.style.position = 'relative';
    notification.style.marginBottom = '10px';
    
    // Create content
    const content = document.createElement('div');
    
    if (title) {
      const titleElement = document.createElement('div');
      titleElement.textContent = title;
      titleElement.style.fontWeight = '700';
      titleElement.style.fontSize = '14px';
      titleElement.style.marginBottom = '4px';
      content.appendChild(titleElement);
    }
    
    const messageElement = document.createElement('div');
    messageElement.textContent = message;
    messageElement.style.fontSize = '14px';
    content.appendChild(messageElement);
    
    // Create close button
    const closeButton = document.createElement('button');
    closeButton.textContent = '×';
    closeButton.style.position = 'absolute';
    closeButton.style.top = '8px';
    closeButton.style.right = '8px';
    closeButton.style.background = 'none';
    closeButton.style.border = 'none';
    closeButton.style.cursor = 'pointer';
    closeButton.style.fontSize = '16px';
    closeButton.style.fontWeight = 'bold';
    
    // Add elements to notification
    notification.appendChild(content);
    notification.appendChild(closeButton);
    
    // Add notification to container
    container.appendChild(notification);
    
    // Remove function
    const removeNotification = () => {
      if (notification.parentNode === container) {
        container.removeChild(notification);
      }
    };
    
    // Add event listener to close button
    closeButton.addEventListener('click', removeNotification);
    
    // Auto close
    if (autoClose > 0) {
      setTimeout(removeNotification, autoClose);
    }
  }
}

export const notifications = new NotificationSystem();
