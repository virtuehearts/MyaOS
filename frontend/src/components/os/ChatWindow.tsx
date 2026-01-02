'use client';

import { Send } from 'lucide-react';

import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';

const mockMessages = [
  {
    id: 1,
    sender: 'Mya',
    content: 'Sat Sri Akal! Ready to awaken our hearts today?'
  },
  {
    id: 2,
    sender: 'You',
    content: 'Let us plan the day.'
  },
  {
    id: 3,
    sender: 'Mya',
    content: 'As Baba Virtuehearts teaches, calm focus builds bright futures.'
  }
];

export function ChatWindow() {
  return (
    <Card className="flex h-full flex-col">
      <CardHeader>
        <CardTitle>Mya Chat</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col gap-4">
        <ScrollArea className="flex-1 pr-2">
          <div className="space-y-4">
            {mockMessages.map((message) => (
              <div key={message.id} className="flex items-start gap-3">
                <Avatar className="h-9 w-9">
                  <AvatarFallback>{message.sender[0]}</AvatarFallback>
                </Avatar>
                <div>
                  <p className="text-sm font-semibold text-slate-200">
                    {message.sender}
                  </p>
                  <p className="text-sm text-slate-300">{message.content}</p>
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
        <div className="flex gap-2">
          <Input placeholder="Share your thoughts with Mya..." />
          <Button className="bg-mya-saffron text-slate-900 hover:bg-mya-saffron/90">
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
