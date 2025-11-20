package com.example.chat.repository;

import com.example.chat.domain.entities.Chat;
import org.springframework.data.jpa.repository.EntityGraph;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

public interface ChatRepository extends JpaRepository<Chat, UUID> {
    @EntityGraph(attributePaths = "messages")
    Optional<Chat> findChatWithMessagesById(UUID id);


    @Query("select c from Chat c where c.chatType = com.example.chat.domain.enums.ChatType.GLOBAL   " +
            "or (c.chatType = com.example.chat.domain.enums.ChatType.PRIVATE and c.userId = :userId)")
    List<Chat> findChatsForUser(@Param("userId") UUID userId);

}
